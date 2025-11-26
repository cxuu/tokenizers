use crate::tokenizer::{NormalizedString, Normalizer, Result};
use crate::utils::macro_rules_attribute;

/// Separators that Unicode normalization keeps as is and never combines with a neighbor: they
/// are starters without decomposition that are not the second part of any composition. ASCII
/// letters have the same properties.
fn is_stable_separator(sep: char) -> bool {
    matches!(sep, ' ' | '▁')
}

#[derive(Default, Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct NFD;
impl Normalizer for NFD {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfd();
        Ok(())
    }

    fn map_cut_separator(&self, sep: char) -> Option<char> {
        is_stable_separator(sep).then_some(sep)
    }
}

#[derive(Default, Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct NFKD;
impl Normalizer for NFKD {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfkd();
        Ok(())
    }

    fn map_cut_separator(&self, sep: char) -> Option<char> {
        is_stable_separator(sep).then_some(sep)
    }
}

#[derive(Default, Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct NFC;
impl Normalizer for NFC {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfc();
        Ok(())
    }

    fn map_cut_separator(&self, sep: char) -> Option<char> {
        is_stable_separator(sep).then_some(sep)
    }
}

#[derive(Default, Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct NFKC;
impl Normalizer for NFKC {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfkc();
        Ok(())
    }

    fn map_cut_separator(&self, sep: char) -> Option<char> {
        is_stable_separator(sep).then_some(sep)
    }
}

fn nmt_keeps(c: char) -> bool {
    !matches!(
        c as u32,
        0x0001..=0x0008 |
        0x000B |
        0x000E..=0x001F |
        0x007F |
        0x008F |
        0x009F
    )
}

fn nmt_map(c: char) -> char {
    match c as u32 {
        0x0009 => ' ',
        0x000A => ' ',
        0x000C => ' ',
        0x000D => ' ',
        0x1680 => ' ',
        0x200B..=0x200F => ' ',
        0x2028 => ' ',
        0x2029 => ' ',
        0x2581 => ' ',
        0xFEFF => ' ',
        0xFFFD => ' ',
        _ => c,
    }
}

fn do_nmt(normalized: &mut NormalizedString) {
    // Ascii Control characters
    normalized
        .filter(nmt_keeps)
        // Other code points considered as whitespace.
        .map(nmt_map);
}

#[derive(Default, Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct Nmt;
impl Normalizer for Nmt {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        do_nmt(normalized);
        Ok(())
    }

    fn map_cut_separator(&self, sep: char) -> Option<char> {
        nmt_keeps(sep).then(|| nmt_map(sep))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nfkc() {
        let original = "\u{fb01}".to_string();
        let normalized = "fi".to_string();
        let mut n = NormalizedString::from(original.clone());
        NFKC.normalize(&mut n).unwrap();

        assert_eq!(
            n,
            NormalizedString::new(original, normalized, vec![(0, 3), (0, 3)], 0)
        );

        assert_eq!(n.alignments_original(), vec![(0, 2), (0, 2), (0, 2)]);
    }
}
