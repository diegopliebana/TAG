//! Operating-system entropy for application setup, outside simulation rules.
use std::io::Result;

#[cfg(windows)]
pub fn automatic_seed() -> Result<u64> {
    #[link(name = "bcrypt")]
    unsafe extern "system" {
        fn BCryptGenRandom(
            algorithm: *mut std::ffi::c_void,
            buffer: *mut u8,
            length: u32,
            flags: u32,
        ) -> i32;
    }
    let mut bytes = [0_u8; 8];
    // SAFETY: the writable buffer contains eight bytes, its length matches the
    // argument, and flag 2 requests the system-preferred RNG with a null handle.
    // Windows owns the function; it cannot retain a pointer to this local buffer.
    let status = unsafe { BCryptGenRandom(std::ptr::null_mut(), bytes.as_mut_ptr(), 8, 2) };
    if status < 0 {
        return Err(std::io::Error::other(format!(
            "BCryptGenRandom returned status {status}"
        )));
    }
    Ok(u64::from_le_bytes(bytes))
}
#[cfg(unix)]
pub fn automatic_seed() -> Result<u64> {
    use std::io::Read;
    let mut bytes = [0_u8; 8];
    std::fs::File::open("/dev/urandom").and_then(|mut file| file.read_exact(&mut bytes))?;
    Ok(u64::from_le_bytes(bytes))
}
#[cfg(not(any(windows, unix)))]
pub fn automatic_seed() -> Result<u64> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "this platform must supply a seed source or an explicit seed",
    ))
}
