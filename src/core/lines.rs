//! Visit straight groups of board cells and find consecutive matches.
//!
//! A window is a target-length group of consecutive cells, such as four positions
//! in a Connect 4 diagonal. An evaluator may inspect all such windows; winning-line
//! code may use a different documented order when several lines win at once.

/// Enumerate every straight target-length window in four board directions.
///
/// # Arguments
/// Dimensions and `target` are copied.
///
/// # Returns
/// An owned iterator yielding each window's coordinates on demand.
/// Windows run right, down, diagonally down-right or up-right. Zero targets yield
/// an empty iterator; target one visits each cell once.
/// Every yielded window fits entirely inside the board.
pub fn windows(
    width: usize,  // Copied column count.
    height: usize, // Copied row count.
    target: usize, // Copied window length.
) -> impl Iterator<Item = impl Iterator<Item = (usize, usize)>> {
    let directions = [(1, 0), (0, 1), (1, 1), (1, -1)];
    (0..width).flat_map(move |x| {
        // Visit the cells in this column as possible window starts.
        (0..height).flat_map(move |y| {
            // Try each allowed direction from this starting cell.
            directions
                .into_iter()
                .take(if target == 1 { 1 } else { 4 })
                .filter_map(move |step| {
                    // Keep only windows whose final cell remains inside the board.
                    let distance = target.checked_sub(1)?;
                    let end_x = if step.0 == 0 {
                        Some(x)
                    } else {
                        x.checked_add(distance)
                    }?;
                    let end_y = match step.1 {
                        -1 => y.checked_sub(distance),
                        0 => Some(y),
                        _ => y.checked_add(distance),
                    }?;
                    (end_x < width && end_y < height).then(|| {
                        // Yield this window's cells.
                        ray(width, height, (x, y), step).take(target)
                    })
                })
        })
    })
}

/// Visit coordinates along a straight direction until reaching the board edge.
///
/// # Arguments
/// `width`, `height`, `start=(x,y)` and `step=(dx,dy)` are copied.
/// A step such as (1,-1) moves right and upward.
///
/// # Returns
/// An owned, cloneable iterator yielding copied coordinates including the start. Invalid starts or a zero step produce an empty iterator; coordinate overflow ends traversal.
pub fn ray(
    width: usize,
    height: usize,
    start: (usize, usize),
    step: (isize, isize),
) -> impl Iterator<Item = (usize, usize)> + Clone {
    // Copied dimensions/coordinates; returns an owned iterator.
    let first = (start.0 < width && start.1 < height && step != (0, 0)).then_some(start);
    std::iter::successors(first, move |&(x, y)| {
        // Advance to the next cell in this direction, stopping at the board edge.
        let next = (x.checked_add_signed(step.0)?, y.checked_add_signed(step.1)?);
        (next.0 < width && next.1 < height).then_some(next)
    })
}

/// Find the first target-length sequence of consecutive matching coordinates.
///
/// # Arguments
/// `coordinates` is consumed and cloned to retain traversal order. Its clone must replay the same coordinates.
/// `target` is the copied required length; zero returns `None`.
/// `matches` is a consumed predicate called on each visited coordinate; false breaks the run.
///
/// # Returns
/// `Some(Vec<(usize,usize)>)` owns the first matching target cells in traversal order.
/// `None` means the traversal finished without a match.
pub fn first_matching_run<I, P>(
    coordinates: I,
    target: usize,
    mut matches: P,
) -> Option<Vec<(usize, usize)>>
where
    I: Iterator<Item = (usize, usize)> + Clone,
    P: FnMut((usize, usize)) -> bool,
{
    // Consumes the coordinate iterator/predicate; returns owned winning coordinates.
    // Clone copies traversal progress; allocate coordinates when a run is found.
    if target == 0 {
        return None;
    }
    let original = coordinates.clone();
    let mut count = 0;
    for (index, cell) in coordinates.enumerate() {
        if matches(cell) {
            count += 1;
            if count == target {
                return Some(original.skip(index + 1 - target).take(target).collect());
            }
        } else {
            // An empty or different cell breaks this consecutive sequence.
            count = 0;
        }
    }
    None
}
