{-# LANGUAGE GADTs #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE DataKinds, TypeSynonymInstances, TypeFamilies, TypeOperators #-}
{-# LANGUAGE UndecidableInstances, FlexibleInstances #-}

module PetriNet where

import qualified Data.Set as Set
import Data.Maybe
--import Data.Trie

data Nat = Z | S Nat

convert :: Int -> Nat
convert x
          | x <= 0 = Z
          | otherwise = S (convert (x-1))

-- A List of length 'n' holding values of type 'a'
data List n a where
    Nil  :: List Z a
    Cons :: a -> List m a -> List (S m) a

-- Just for visualization (a better instance would work with read)
instance Show a => Show (List n a) where
    show Nil = "Nil"
    show (Cons x xs) = show x ++ "-" ++ show xs

--do the function f on the list of a's and b's to get a list of c's
g :: (a -> b -> c) -> List n a -> List n b -> List n c
g f (Cons x xs) (Cons y ys) = Cons (f x y) $ g f xs ys
g f Nil Nil = Nil

-- adding vectors of length n
listAdd :: (Num a) => List n a -> List n a -> List n a
listAdd = g (\x y -> x+y)
--subtracting vectors of length n
listSub :: (Num a) => List n a -> List n a -> List n a
listSub = g (\x y -> x-y)
-- and of all the booleans in this list
andAll :: List n Bool -> Bool
andAll Nil = True
andAll (Cons x xs) = x && (andAll xs) 
-- are all the entries of this vector nonnegative
allPositive :: (Num a,Ord a) => List n a -> Bool
allPositive Nil = True
allPositive (Cons x xs) = (x>=0) && allPositive xs
-- extract the m'th element of the list, if m<0 or m too big then outputs Nothing
getElement :: Int -> List n a -> Maybe a
getElement 0 (Cons x _) = Just x
getElement m (Cons x xs) = getElement (m-1) xs
getElement _ _ = Nothing

type Matrix n m a = List n (List m a)
-- same as g but on something that is n by m instead of just a 1 dimensional list
g2 :: (a -> b -> c) -> (Matrix n m a) -> (Matrix n m b) -> (Matrix n m c)
g2 f = g (g f)

data Bounds n = List n Int :< List n Int

-- all occ_i should be between l_i and u_i for the list of occupationNumbers, and lists of lower and upper Bounds
withinBounds :: List numPlaces Int -> Bounds numPlaces -> Bool
withinBounds occupationNumbers (lowerBounds :< upperBounds) = andAll (g (\x y -> x<=y) lowerBounds occupationNumbers) && andAll (g (\x y -> x<=y) occupationNumbers upperBounds)

data PetriNet n t = PetriNet{wPlus :: Matrix t n Int, wMinus :: Matrix t n Int, occupationNumbers :: List n Int, placeCapacities :: Bounds n}

-- a transition is fired in two steps, one is the input into the transition state
-- the second is the output from intermediate transition state into the outputs
-- this is done to avoid the possibility of needing the outputs of the transition to be able to fire
-- like X -> 2X but starting with 0 X. Could fire if had 0X -> -1 X -> 1X capability but that would require integers instead of natural numbers
-- when invalid get Nothing by using the Maybe monad here
fireTransition :: Maybe (PetriNet n t) -> Int -> Maybe (PetriNet n t)
fireTransitionPart1 :: Maybe (PetriNet n t) -> Int -> Maybe (PetriNet n t)
fireTransitionPart2 :: Maybe (PetriNet n t) -> Int -> Maybe (PetriNet n t)

fireTransitionPart1 Nothing _ = Nothing
fireTransitionPart1 (Just myPetriNet) myTransition
                                       | withinBounds (occupationNumbers newPetriNet) (placeCapacities newPetriNet)= Just newPetriNet
                                       | otherwise = Nothing
                                       where newPetriNet=PetriNet{wPlus=wPlus myPetriNet,wMinus=wMinus myPetriNet,
                                       occupationNumbers = listSub (occupationNumbers myPetriNet) (fromJust $ getElement myTransition (wMinus myPetriNet)),
                                       placeCapacities=placeCapacities myPetriNet}
fireTransitionPart2 myPetriNet myTransition
                                       | isNothing myPetriNet = Nothing
                                       | withinBounds (occupationNumbers newPetriNet) (placeCapacities newPetriNet)= Just newPetriNet
                                       | otherwise = Nothing
                                       where newPetriNet=PetriNet{wPlus=wPlus $ fromJust myPetriNet,wMinus=wMinus $ fromJust myPetriNet,
                                       occupationNumbers = listAdd (occupationNumbers $ fromJust myPetriNet) (fromJust $ getElement myTransition $ wPlus $ fromJust myPetriNet ),
                                       placeCapacities=placeCapacities $ fromJust myPetriNet}
fireTransition myPetriNet myTransition = fireTransitionPart2 (fireTransitionPart1 myPetriNet myTransition) myTransition

wPlusT1Ex = (Cons 0 $ Cons 1 $ Cons 1 $ Cons 0 Nil)
wPlusT2Ex = (Cons 1 $ Cons 0 $ Cons 0 $ Cons 1 Nil)
wPlusEx = Cons wPlusT1Ex (Cons wPlusT2Ex Nil)
wMinusT1Ex = (Cons 1 $ Cons 0 $ Cons 0 $ Cons 0 Nil)
wMinusT2Ex = (Cons 0 $ Cons 1 $ Cons 1 $ Cons 0 Nil)
wMinusEx = Cons wMinusT1Ex (Cons wMinusT2Ex Nil)
occupationNumbersEx = Cons 1 $ Cons 0 $ Cons 2 $ Cons 1 Nil
boundsEx = (Cons 0 $ Cons 0 $ Cons 0 $ Cons 0 Nil) :< (Cons 10 $ Cons 10 $ Cons 10 $ Cons 10 Nil)
petriNetEx = PetriNet{wPlus=wPlusEx,wMinus=wMinusEx,occupationNumbers=occupationNumbersEx,placeCapacities=boundsEx}
-- t0 can fire but t1 cannot, so new1Ex is okay but new2Ex is Nothing
new1Ex = fireTransition (Just petriNetEx) 0
new2Ex = fireTransition (Just petriNetEx) 1

-- store a prefix tree, where the paths are prefixes. The information stored at the vertices is
-- the bounds on the incoming occupationNumbers if that firing sequence is to be sensible and the change in occupationNumbers
-- if that firing sequence were executed
--data Leaf numPlaces = Leaf {myBounds::Bounds numPlaces,overallChange::List numPlaces Int,firable :: Bool}
--data InternalNode numPlaces = InternalNode {myBounds::Bounds numPlaces,overallChange::List numPlaces Int,firable::Bool}
--data MyPrefixTree numPlaces numTransitions = Leaf numPlaces | (InternalNode numPlaces,List numTransitions (MyPrefixTree numPlaces numTransitions))

-- x'th child in the trie where 0<=x<numTransitions if this is not possible because x is outside this range
-- or if trying to do children of a leaf, then gives Nothing
--subTrie :: Int -> MyPrefixTree numPlaces numTransitions -> Maybe MyPrefixTree numPlaces numTransitions
--subTrie _ Leaf{myBounds=_} = Nothing
--subTrie x (_,ls) = getElement x ls

--go to the vertex of the trie and extract it's data. Nothing if failure because said vertex does not exist.
--getBoundsWord :: Maybe (MyPrefixTree numPlaces numTransitions) -> [Int] -> Maybe (Bounds numPlaces)
--getBoundsWord (Just Leaf{myBounds=x,overallChange=_,firable=_}) [] = Just x
--getBoundsWord (Just Leaf{myBounds=_,overallChange=_,firable=_}) _ = Nothing
--getBoundsWord (Just (InternalNode{myBounds=x,overallChange=_,firable=_},_)) [] = Just x
--getBoundsWord (Just (_,y)) x:xs = getBoundsWord (subTrie x y) 
--getOverallChangeWord :: Maybe (MyPrefixTree numPlaces numTransitions) -> [Int] -> Maybe (List numPlaces Int)
--getOverallChangeWord (Just Leaf{myBounds=_,overallChange=x,,firable=_}) [] = Just x
--getOverallChangeWord (Just Leaf{myBounds=_,overallChange=_,firable=_}) _ = Nothing
--getOverallChangeWord (Just (InternalNode{myBounds=_,overallChange=x,firable=_},_)) [] = Just x
--getOverallChangeWord (Just (_,y)) x:xs = getBoundsWord (subTrie x y) 
--getFirable :: Maybe (MyPrefixTree numPlaces numTransitions) -> [Int] -> Maybe Bool
--getFirable (Just Leaf{myBounds=_,overallChange=_,firable=x}) [] = Just x
--getFirable (Just Leaf{myBounds=_,overallChange=_,firable=_}) _ = Nothing
--getFirable (Just (InternalNode{myBounds=_,overallChange=_,firable=x},_)) [] = Just x
--getFirable (Just (_,y)) x:xs = getBoundsWord (subTrie x y) 