//! Small owned-card operations shared by games; visibility and effects belong to each game.
use super::random::ChanceRng;
use std::collections::VecDeque;

/// Shuffle an owned sequence with the supplied chance stream.
/// Each position receives a uniform index from its remaining prefix.
/// The caller owns the RNG, so the shuffle advances the real game or simulation
/// stream it explicitly supplied; an empty or one-item slice consumes nothing.
pub fn shuffle<T>(values: &mut [T], random: &mut ChanceRng) {
    for i in (1..values.len()).rev() {
        let j = random.index(i + 1).expect("positive shuffle bound");
        values.swap(i, j);
    }
}

/// Compare card supplies including the number of copies of each value.
pub fn same_multiset<C: Clone + Ord>(expected: &[C], supplied: &[C]) -> bool {
    if expected.len() != supplied.len() {
        return false;
    }
    let mut expected = expected.to_vec();
    let mut supplied = supplied.to_vec();
    expected.sort();
    supplied.sort();
    expected == supplied
}

/// Validate an optional controlled order, consume a normal shuffle, then use
/// that order if present. An invalid order leaves `random` unchanged.
pub fn prepare_deck<C: Clone + Ord>(
    standard: &[C],
    supplied: Option<&[C]>,
    random: &mut ChanceRng,
) -> Result<Vec<C>, InvalidDeck> {
    if supplied.is_some_and(|order| !same_multiset(standard, order)) {
        return Err(InvalidDeck);
    }
    let mut deck = standard.to_vec();
    shuffle(&mut deck, random);
    if let Some(order) = supplied {
        deck.clone_from_slice(order);
    }
    Ok(deck)
}

/// The supplied order has a different length, card value or copy count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidDeck;

/// Remove one matching card while retaining ownership of the remaining zone.
pub fn take_one<C: PartialEq>(zone: &mut Vec<C>, card: &C) -> Option<C> {
    let index = zone.iter().position(|candidate| candidate == card)?;
    Some(zone.remove(index))
}

/// Remove each requested owned card from a temporary supply. On failure,
/// return the first missing card; earlier removals remain in `zone`.
pub fn remove_all<C: PartialEq>(
    zone: &mut Vec<C>,
    requested: impl IntoIterator<Item = C>,
) -> Result<(), C> {
    for card in requested {
        if take_one(zone, &card).is_none() {
            return Err(card);
        }
    }
    Ok(())
}

/// Draw the next card from a queue, transferring ownership to the caller.
pub fn draw<C>(pile: &mut VecDeque<C>) -> Option<C> {
    pile.pop_front()
}

/// Move all cards from one owned zone into another, preserving order.
pub fn transfer_all<C>(source: &mut Vec<C>, target: &mut Vec<C>) {
    target.append(source);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::checkpoint::Checkpoint;

    #[test]
    fn controlled_decks_validate_counts_before_advancing_chance() {
        let standard = [1, 1, 2, 3];
        let mut random = ChanceRng::new(41);
        let before = random.snapshot();
        assert_eq!(
            prepare_deck(&standard, Some(&[1, 2, 2, 3]), &mut random),
            Err(InvalidDeck)
        );
        assert_eq!(random.snapshot(), before);
        let controlled = prepare_deck(&standard, Some(&[3, 1, 2, 1]), &mut random).unwrap();
        assert_eq!(controlled, [3, 1, 2, 1]);
        let mut ordinary = ChanceRng::new(41);
        prepare_deck(&standard, None, &mut ordinary).unwrap();
        assert_eq!(random.snapshot(), ordinary.snapshot());
    }

    #[test]
    fn transfers_only_one_matching_copy_and_preserves_independent_clones() {
        let mut hand = vec![1, 2, 1];
        let copy = hand.clone();
        assert_eq!(take_one(&mut hand, &1), Some(1));
        assert_eq!(hand, [2, 1]);
        let mut pile = VecDeque::from([3]);
        assert_eq!(draw(&mut pile), Some(3));
        assert_eq!(remove_all(&mut hand, [1, 2]), Ok(()));
        assert!(hand.is_empty());
        let mut supply = vec![1, 2];
        assert_eq!(remove_all(&mut supply, [1, 1]), Err(1));
        assert_eq!(supply, [2]);
        hand = vec![2, 1];
        transfer_all(&mut hand, &mut Vec::new());
        assert!(hand.is_empty());
        assert_eq!(copy, [1, 2, 1]);
    }
}
