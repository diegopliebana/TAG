use tag_rust::core::PlayerId;

pub(crate) fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}

// Independent test oracle: construct each window directly from signed coordinates.
pub(crate) fn oracle(
    width: usize,
    height: usize,
    target: usize,
    cells: &[Option<PlayerId>],
    player: PlayerId,
) -> f64 {
    let mut values = Vec::new();
    for x in 0..width {
        for y in 0..height {
            for (dx, dy) in [(1isize, 0isize), (0, 1), (1, 1), (1, -1)] {
                if target == 1 && (dx, dy) != (1, 0) {
                    continue;
                }
                let mut seats = Vec::new();
                for offset in 0..target {
                    let cx = x as isize + offset as isize * dx;
                    let cy = y as isize + offset as isize * dy;
                    if cx < 0 || cy < 0 || cx >= width as isize || cy >= height as isize {
                        break;
                    }
                    seats.push(cells[cx as usize * height + cy as usize]);
                }
                if seats.len() != target {
                    continue;
                }
                let mut friendly = 0;
                let mut opposing = 0;
                for seat in seats.into_iter().flatten() {
                    if seat == player {
                        friendly += 1;
                    } else {
                        opposing += 1;
                    }
                }
                let value = match (friendly, opposing) {
                    (_, 0) => (friendly as f64 / target as f64).powi(2),
                    (0, _) => -(opposing as f64 / target as f64).powi(2),
                    _ => 0.0,
                };
                values.push(value);
            }
        }
    }
    if values.is_empty() {
        0.0
    } else {
        0.9 * values.iter().sum::<f64>() / values.len() as f64
    }
}
