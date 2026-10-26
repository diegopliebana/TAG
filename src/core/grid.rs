//! Rectangular cell storage, with dimension and allocation checks.
//!
//! Cells are stored column by column: index=x*height+y. This module does not decide
//! which moves are legal or which cells a player may see; those are game rules.
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)] // Clone/Copy: one count; PartialEq/Eq: exact reflexive limit equality.
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(deny_unknown_fields))]
pub struct BoardLimits {
    pub max_cells: usize,
}

impl Default for BoardLimits {
    fn default() -> Self {
        // No borrowed input; returns a copied configuration value.
        Self {
            max_cells: 1_000_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)] // Clone: explicit error duplication.
// PartialEq/Eq: compare error details; all fields have reflexive equality.
pub enum GridError {
    InvalidDimensions {
        width: usize,
        height: usize,
    },
    InvalidCellLimit,
    SizeOverflow,
    CellLimitExceeded {
        requested: usize,
        limit: usize,
    },
    AllocationFailed,
    OutOfBounds {
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    },
}

impl fmt::Display for GridError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Shared error borrow, exclusive output borrow; returns formatting success/error.
        match self {
            Self::InvalidDimensions { width, height } => write!(
                formatter,
                "board dimensions must be positive, received {width}×{height}"
            ),
            Self::InvalidCellLimit => {
                write!(formatter, "maximum board cell count must be positive")
            }
            Self::SizeOverflow => write!(
                formatter,
                "board dimensions overflow the supported cell count"
            ),
            Self::CellLimitExceeded { requested, limit } => write!(
                formatter,
                "board needs {requested} cells, exceeding configured limit {limit}"
            ),
            Self::AllocationFailed => write!(formatter, "could not allocate board storage"),
            Self::OutOfBounds {
                x,
                y,
                width,
                height,
            } => write!(
                formatter,
                "cell ({x}, {y}) is outside the {width}×{height} board"
            ),
        }
    }
}
impl std::error::Error for GridError {}

/// Flat x-major storage: index = x * height + y. All fields are private.
#[derive(Clone, Debug, PartialEq, Eq)] // Clone: explicit owned vector copy.
// PartialEq/Eq: dimensions and cells compare when T supports them; isolation also depends on T's Clone.
pub struct Grid<T> {
    width: usize,
    height: usize,
    cells: Vec<T>,
}

impl<T> Grid<T> {
    /// Construct an owned rectangular grid filled with copies of one value.
    ///
    /// # Arguments
    /// `width` and `height` are copied positive dimensions.
    /// `value` is consumed and cloned to populate the cells.
    /// `limits` is a copied `BoardLimits` policy for this grid.
    ///
    /// # Returns
    /// `Ok(Grid<T>)` owns its cells.
    ///
    /// # Errors
    /// Zero dimensions/limit, size overflow, exceeded cell limit or failed initial reservation return `GridError`. Later clones can still allocate.
    pub fn filled(
        width: usize,        // Copied dimension.
        height: usize,       // Copied dimension.
        value: T,            // Owned value consumed; cloned to populate the board.
        limits: BoardLimits, // Copied resource policy.
    ) -> Result<Self, GridError>
    where
        T: Clone,
    {
        // Returns owned grid or error.
        if width == 0 || height == 0 {
            return Err(GridError::InvalidDimensions { width, height });
        }
        if limits.max_cells == 0 {
            return Err(GridError::InvalidCellLimit);
        }
        let count = width.checked_mul(height).ok_or(GridError::SizeOverflow)?;
        if count > limits.max_cells {
            return Err(GridError::CellLimitExceeded {
                requested: count,
                limit: limits.max_cells,
            });
        }
        let mut cells = Vec::new();
        cells.try_reserve_exact(count).map_err(|_| {
            // Report inability to reserve this board's storage as a construction error.
            GridError::AllocationFailed
        })?;
        cells.resize(count, value);
        Ok(Self {
            width,
            height,
            cells,
        })
    }

    /// Return the copied `usize` number of board columns.
    pub fn width(&self) -> usize {
        // Shared borrow; copied dimension out.
        self.width
    }
    /// Return the copied `usize` number of board rows.
    pub fn height(&self) -> usize {
        // Shared borrow; copied dimension out.
        self.height
    }
    /// Return a shared borrowed `&[T]` slice of all cells, indexed by x * height + y.
    pub fn cells(&self) -> &[T] {
        // Shared borrow; borrowed slice tied to self.
        &self.cells
    }
    /// Convert copied zero-based `x`/`y` coordinates to a flat x-major `usize` index.
    /// Returns an owned `GridError::OutOfBounds` if either coordinate is outside this borrowed grid.
    fn index(&self, x: usize, y: usize) -> Result<usize, GridError> {
        // Shared borrow and copied coordinates; copied index or owned error out.
        if x >= self.width || y >= self.height {
            return Err(GridError::OutOfBounds {
                x,
                y,
                width: self.width,
                height: self.height,
            });
        }
        Ok(x * self.height + y) // Constructor validated index arithmetic.
    }
    /// Borrow one cell for reading.
    ///
    /// # Arguments
    /// `x` and `y` are copied zero-based coordinates; `self` is passed by a shared (read-only) reference.
    ///
    /// # Returns
    /// `Ok(&T)` borrows the cell for as long as the grid borrow is valid.
    ///
    /// # Errors
    /// `GridError::OutOfBounds` if either coordinate is outside the grid.
    pub fn get(&self, x: usize, y: usize) -> Result<&T, GridError> {
        // Shared grid borrow, copied coordinates; returns borrowed cell or owned error.
        Ok(&self.cells[self.index(x, y)?])
    }
    /// Borrow one cell exclusively for modification.
    ///
    /// # Arguments
    /// `x` and `y` are copied zero-based coordinates; `self` is accessed through an exclusive mutable reference.
    ///
    /// # Returns
    /// `Ok(&mut T)` borrows exclusive mutable access to the cell.
    ///
    /// # Errors
    /// `GridError::OutOfBounds` if either coordinate is outside the grid.
    pub fn get_mut(&mut self, x: usize, y: usize) -> Result<&mut T, GridError> {
        // Exclusive grid borrow, copied coordinates; returns exclusive cell borrow.
        let index = self.index(x, y)?;
        Ok(&mut self.cells[index])
    }
    /// Borrow the grid and return an iterator of copied `(x, y)` coordinates, columns first then rows.
    pub fn coordinates(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        // Shared grid borrow; returns an iterator borrowing dimensions, yielding copied coordinates.
        (0..self.width).flat_map(|x| {
            // Visit each row in this column, preserving x-major order.
            (0..self.height).map(move |y| {
                // Return this column and row's cell coordinates.
                (x, y)
            })
        })
    }
}
