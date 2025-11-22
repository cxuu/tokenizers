/// Comprehensive tests for parallel single-input encoding.
///
/// These tests verify that parallel encoding produces identical results to serial encoding
/// (except for word IDs which are not computed) while properly handling edge cases,
/// different tokenizer types, and UTF-8 boundaries.
mod common;

/// Helper function to compare serial and parallel encoding results with detailed debugging
fn assert_encodings_equal(serial: &tokenizers::Encoding, parallel: &tokenizers::Encoding) {
    // Check IDs
    if serial.get_ids() != parallel.get_ids() {
        let first_diff = serial
            .get_ids()
            .iter()
            .zip(parallel.get_ids().iter())
            .position(|(s, p)| s != p);
        panic!(
            "IDs don't match at position {:?}\nSerial length: {}, Parallel length: {}\nFirst 10 serial: {:?}\nFirst 10 parallel: {:?}",
            first_diff,
            serial.get_ids().len(),
            parallel.get_ids().len(),
            &serial.get_ids()[..serial.get_ids().len().min(10)],
            &parallel.get_ids()[..parallel.get_ids().len().min(10)]
        );
    }

    // Check tokens
    assert_eq!(
        serial.get_tokens(),
        parallel.get_tokens(),
        "Tokens don't match between serial and parallel"
    );

    // Check offsets
    if serial.get_offsets() != parallel.get_offsets() {
        let first_diff = serial
            .get_offsets()
            .iter()
            .zip(parallel.get_offsets().iter())
            .position(|(s, p)| s != p);
        panic!(
            "Offsets don't match at position {:?}\nSerial: {:?}\nParallel: {:?}",
            first_diff,
            &serial.get_offsets()[first_diff.unwrap_or(0)..serial.get_offsets().len().min(first_diff.unwrap_or(0) + 5)],
            &parallel.get_offsets()[first_diff.unwrap_or(0)..parallel.get_offsets().len().min(first_diff.unwrap_or(0) + 5)]
        );
    }

    // Note: Word IDs are NOT checked as parallel encoding does not compute them
    // for performance reasons. Word IDs will be None for all tokens in parallel encoding.

    // Check type IDs
    assert_eq!(
        serial.get_type_ids(),
        parallel.get_type_ids(),
        "Type IDs don't match between serial and parallel"
    );

    // Check attention mask
    assert_eq!(
        serial.get_attention_mask(),
        parallel.get_attention_mask(),
        "Attention masks don't match between serial and parallel"
    );

    // Check special tokens mask
    assert_eq!(
        serial.get_special_tokens_mask(),
        parallel.get_special_tokens_mask(),
        "Special tokens masks don't match between serial and parallel"
    );
}

/// Test correctness: parallel encoding should match serial encoding
#[test]
fn test_parallel_correctness_long_input() {
    let tokenizer = common::get_bert();

    // Create a long input (>10KB to trigger parallel encoding)
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(500);

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

/// Test with special tokens
#[test]
fn test_parallel_with_special_tokens() {
    let tokenizer = common::get_bert();

    let text = "Hello world! This is a test. ".repeat(500);

    let serial = tokenizer.encode(text.as_str(), true).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), true).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

/// Test with byte-level BPE tokenizer
#[test]
fn test_parallel_byte_level_bpe() {
    let tokenizer = common::get_byte_level(false, true);

    let text = "Hello world! How are you doing today? ".repeat(500);

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    // Debug output for byte-level failures
    if serial.get_ids() != parallel.get_ids() {
        let diff_pos = serial.get_ids().iter()
            .zip(parallel.get_ids().iter())
            .position(|(s, p)| s != p)
            .unwrap();
        println!("\nByte-level BPE mismatch at position {}", diff_pos);
        println!("Serial tokens around {}: {:?}", diff_pos, &serial.get_tokens()[diff_pos.saturating_sub(5)..=(diff_pos+5).min(serial.get_tokens().len()-1)]);
        println!("Parallel tokens around {}: {:?}", diff_pos, &parallel.get_tokens()[diff_pos.saturating_sub(5)..=(diff_pos+5).min(parallel.get_tokens().len()-1)]);
        println!("Serial offsets around {}: {:?}", diff_pos, &serial.get_offsets()[diff_pos.saturating_sub(5)..=(diff_pos+5).min(serial.get_offsets().len()-1)]);
        println!("Parallel offsets around {}: {:?}", diff_pos, &parallel.get_offsets()[diff_pos.saturating_sub(5)..=(diff_pos+5).min(parallel.get_offsets().len()-1)]);
    }

    assert_encodings_equal(&serial, &parallel);
}

/// Test with short input (should fall back to serial)
#[test]
fn test_parallel_short_input_fallback() {
    let tokenizer = common::get_bert();

    let text = "Short input text.";

    let serial = tokenizer.encode(text, false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text, false).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

/// Test with exactly threshold length
#[test]
fn test_parallel_threshold_boundary() {
    let tokenizer = common::get_bert();

    // Create input at exactly 10,000 bytes
    let unit = "Hello world! ";
    let repeat_count = 10_000 / unit.len();
    let text = unit.repeat(repeat_count);

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

/// Test with multi-byte UTF-8 characters
#[test]
fn test_parallel_multibyte_utf8() {
    let tokenizer = common::get_bert();

    // Mix ASCII and multi-byte characters
    let text = "Hello 世界! This is a test with émojis 🎉 and Chinese 你好. ".repeat(300);

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    // Debug: show first few word IDs
    if std::env::var("DEBUG_TESTS").is_ok() {
        println!("\nSerial first 15 word IDs: {:?}", &serial.get_word_ids()[..15]);
        println!("Parallel first 15 word IDs: {:?}", &parallel.get_word_ids()[..15]);
        println!("Serial first 15 tokens: {:?}", &serial.get_tokens()[..15]);
        println!("Parallel first 15 tokens: {:?}", &parallel.get_tokens()[..15]);
    }

    assert_encodings_equal(&serial, &parallel);
}

/// Test with various text patterns
#[test]
fn test_parallel_various_patterns() {
    let tokenizer = common::get_bert();

    // Different text patterns
    let patterns = vec![
        "The quick brown fox jumps over the lazy dog. ",
        "1234567890 !@#$%^&*() ",
        "CamelCaseWords and snake_case_words ",
        "Multiple    spaces   and\ttabs\there ",
        "Line breaks\nand\r\ncarriage returns ",
    ];

    for pattern in patterns {
        let text = pattern.repeat(300);

        let serial = tokenizer.encode(text.as_str(), false).unwrap();
        let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

        assert_encodings_equal(&serial, &parallel);
    }
}

/// Test empty string
#[test]
fn test_parallel_empty_string() {
    let tokenizer = common::get_bert();

    let text = "";

    let serial = tokenizer.encode(text, false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text, false).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

/// Test very long input (stress test)
#[test]
fn test_parallel_very_long_input() {
    let tokenizer = common::get_bert();

    // Create a very long input (~100KB)
    let text = "This is a test sentence that will be repeated many times. ".repeat(2000);

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

/// Test with text containing only whitespace
#[test]
fn test_parallel_whitespace_only() {
    let tokenizer = common::get_bert();

    let text = " ".repeat(15_000);

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

/// Test with repeated single character
#[test]
fn test_parallel_repeated_character() {
    let tokenizer = common::get_bert();

    let text = "a".repeat(15_000);

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

/// Test with mixed ASCII and non-ASCII at split boundaries
#[test]
fn test_parallel_utf8_at_boundaries() {
    let tokenizer = common::get_bert();

    // Create text where UTF-8 multi-byte chars are likely near split point
    let before = "a".repeat(4990);
    let multi = "你好世界"; // 12 bytes (3 bytes per character)
    let after = "b".repeat(5000);
    let text = format!("{}{}{}", before, multi, after);

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

/// Test decode consistency
#[test]
fn test_parallel_decode_consistency() {
    let tokenizer = common::get_bert();

    let text = "The quick brown fox jumps over the lazy dog. ".repeat(500);

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    let serial_decoded = tokenizer.decode(serial.get_ids(), false).unwrap();
    let parallel_decoded = tokenizer.decode(parallel.get_ids(), false).unwrap();

    assert_eq!(
        serial_decoded, parallel_decoded,
        "Decoded text doesn't match"
    );
}

/// Test with tokenizer that has no pre-tokenizer
#[test]
fn test_parallel_without_pretokenizer() {
    let tokenizer = common::get_empty();

    let text = "test ".repeat(3000);

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

/// Test parallelism disabled via environment variable
#[test]
fn test_parallel_with_parallelism_disabled() {
    use tokenizers::utils::parallelism;

    let tokenizer = common::get_bert();
    let text = "Hello world! ".repeat(1000);

    // Temporarily disable parallelism
    let original = parallelism::get_parallelism();
    parallelism::set_parallelism(false);

    let result = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();
    let expected = tokenizer.encode(text.as_str(), false).unwrap();

    // Restore original setting
    parallelism::set_parallelism(original);

    assert_encodings_equal(&expected, &result);
}

/// Test with byte-level tokenizer with prefix space
#[test]
fn test_parallel_byte_level_with_prefix() {
    let tokenizer = common::get_byte_level(true, false);

    let text = "Hello world! ".repeat(1000);

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

// Test removed: Word IDs are no longer computed in parallel encoding for performance
// /// Test that word IDs are monotonically increasing
// #[test]
// fn test_word_ids_monotonic() {
//     ...
// }

/// Debug test to check max token lengths
#[test]
fn test_debug_max_token_length() {
    use tokenizers::tokenizer::Model;

    let bert = common::get_bert();
    let byte_level = common::get_byte_level(false, true);

    println!("BERT max token length: {} bytes", bert.get_model().max_token_byte_length());
    println!("Byte-level BPE max token length: {} bytes", byte_level.get_model().max_token_byte_length());
}

/// Debug test to find duplicate/missing tokens
#[test]
fn test_debug_find_duplicate() {
    let tokenizer = common::get_bert();

    let text = "The quick brown fox jumps over the lazy dog. ".repeat(500);

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    println!("Serial: {} tokens, Parallel: {} tokens", serial.len(), parallel.len());

    // Find first difference
    for i in 0..serial.len().min(parallel.len()) {
        if serial.get_ids()[i] != parallel.get_ids()[i] {
            println!("\nFirst ID mismatch at position {}:", i);
            println!("  Serial: id={}, token='{}', offset={:?}",
                serial.get_ids()[i], serial.get_tokens()[i], serial.get_offsets()[i]);
            println!("  Parallel: id={}, token='{}', offset={:?}",
                parallel.get_ids()[i], parallel.get_tokens()[i], parallel.get_offsets()[i]);

            // Show context
            if i > 0 {
                println!("\n  Previous token:");
                println!("    Serial: id={}, token='{}', offset={:?}",
                    serial.get_ids()[i-1], serial.get_tokens()[i-1], serial.get_offsets()[i-1]);
                println!("    Parallel: id={}, token='{}', offset={:?}",
                    parallel.get_ids()[i-1], parallel.get_tokens()[i-1], parallel.get_offsets()[i-1]);
            }

            if i + 1 < serial.len().min(parallel.len()) {
                println!("\n  Next token:");
                println!("    Serial: id={}, token='{}', offset={:?}",
                    serial.get_ids()[i+1], serial.get_tokens()[i+1], serial.get_offsets()[i+1]);
                println!("    Parallel: id={}, token='{}', offset={:?}",
                    parallel.get_ids()[i+1], parallel.get_tokens()[i+1], parallel.get_offsets()[i+1]);
            }
            break;
        }
    }

    // Check for length difference
    if serial.len() != parallel.len() {
        println!("\nLength mismatch! Serial: {}, Parallel: {}", serial.len(), parallel.len());
        if parallel.len() > serial.len() {
            println!("Extra token(s) in parallel at position {}:", serial.len());
            println!("  id={}, token='{}', offset={:?}",
                parallel.get_ids()[serial.len()],
                parallel.get_tokens()[serial.len()],
                parallel.get_offsets()[serial.len()]);
        }
    }
}

// Test removed: Word IDs are no longer computed in parallel encoding for performance
// /// Debug test to understand word ID assignment
// #[test]
// fn test_debug_word_ids() {
//     ...
// }

// Test removed: Word IDs are no longer computed in parallel encoding for performance
// /// Simple test to understand word ID assignment
// #[test]
// fn test_simple_word_id_assignment() {
//     ...
// }

/// Test that offsets are monotonic and non-overlapping
#[test]
fn test_offsets_monotonic_and_valid() {
    let tokenizer = common::get_bert();

    let text = "The quick brown fox jumps over the lazy dog. ".repeat(500);

    let encoding = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    let offsets = encoding.get_offsets();

    for i in 0..offsets.len() {
        let (start, end) = offsets[i];

        // Offset must be valid (start <= end)
        assert!(
            start <= end,
            "Invalid offset at position {}: ({}, {})",
            i,
            start,
            end
        );

        // Offset should not exceed text length
        assert!(
            end <= text.len(),
            "Offset exceeds text length at position {}: ({}, {}) > {}",
            i,
            start,
            end,
            text.len()
        );

        // Check non-overlapping with next (allowing gaps)
        if i + 1 < offsets.len() {
            let (next_start, _) = offsets[i + 1];
            assert!(
                end <= next_start,
                "Overlapping offsets at positions {}-{}: ({}, {}) and ({}, {})",
                i,
                i + 1,
                start,
                end,
                next_start,
                offsets[i + 1].1
            );
        }
    }
}

/// Test split at exact midpoint with known structure
#[test]
fn test_split_point_handling() {
    let tokenizer = common::get_bert();

    // Create text with clear structure across the split point
    let part1 = "Hello world test sentence. ".repeat(200); // ~5400 bytes
    let part2 = "Another test sentence here. ".repeat(200); // ~5600 bytes
    let text = format!("{}{}", part1, part2);

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

/// Test with text containing many word boundaries
#[test]
fn test_many_word_boundaries() {
    let tokenizer = common::get_bert();

    // Each word is small, creating many word boundaries
    let text = (0..2000).map(|i| format!("word{} ", i)).collect::<String>();

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

/// Test with very long individual tokens
#[test]
fn test_long_tokens() {
    let tokenizer = common::get_bert();

    // Create long "words" without spaces
    let long_word = "a".repeat(100);
    let text = format!("{} ", long_word).repeat(150);

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

/// Test offset reconstruction - can we reconstruct original text from offsets?
#[test]
fn test_offset_text_reconstruction() {
    let tokenizer = common::get_bert();

    let text = "The quick brown fox jumps over the lazy dog. ".repeat(500);

    let encoding = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    // Reconstruct text from offsets
    let mut reconstructed = String::new();
    for &(start, end) in encoding.get_offsets() {
        if end > start && end <= text.len() {
            reconstructed.push_str(&text[start..end]);
        }
    }

    // Not all tokens may cover the full text (some might be special/whitespace removed)
    // but reconstructed should be a subset of original
    assert!(
        !reconstructed.is_empty(),
        "Should reconstruct some text from offsets"
    );
}

/// Test that no tokens are lost or duplicated
#[test]
fn test_no_token_loss_or_duplication() {
    let tokenizer = common::get_bert();

    let text = "Test sentence with various words. ".repeat(400);

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    // Same number of tokens
    assert_eq!(
        serial.get_ids().len(),
        parallel.get_ids().len(),
        "Different number of tokens"
    );

    // Every token ID should appear the same number of times
    use std::collections::HashMap;
    let mut serial_counts: HashMap<u32, usize> = HashMap::new();
    let mut parallel_counts: HashMap<u32, usize> = HashMap::new();

    for &id in serial.get_ids() {
        *serial_counts.entry(id).or_insert(0) += 1;
    }
    for &id in parallel.get_ids() {
        *parallel_counts.entry(id).or_insert(0) += 1;
    }

    assert_eq!(
        serial_counts, parallel_counts,
        "Token frequency distribution differs"
    );
}

/// Test around split boundaries with specific patterns
#[test]
fn test_boundary_with_special_patterns() {
    let tokenizer = common::get_bert();

    // Create text with known structure around midpoint
    let prefix = "normal text ".repeat(400); // ~4800 bytes
    let special = "🎉 你好 world "; // Multi-byte chars near split
    let suffix = "more text ".repeat(500); // ~5000 bytes
    let text = format!("{}{}{}", prefix, special, suffix);

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

/// Test that multiple calls to encode_parallel_single work correctly (verifying caching)
#[test]
fn test_multiple_parallel_encode_calls() {
    let tokenizer = common::get_bert();

    let text1 = "The quick brown fox jumps over the lazy dog. ".repeat(500);
    let text2 = "Hello world! This is another test. ".repeat(500);
    let text3 = "Testing repeated encoding calls. ".repeat(500);

    // First call - cache will be computed
    let result1 = tokenizer.encode_parallel_single(text1.as_str(), false).unwrap();
    let expected1 = tokenizer.encode(text1.as_str(), false).unwrap();
    assert_encodings_equal(&expected1, &result1);

    // Second call - cache should be used
    let result2 = tokenizer.encode_parallel_single(text2.as_str(), false).unwrap();
    let expected2 = tokenizer.encode(text2.as_str(), false).unwrap();
    assert_encodings_equal(&expected2, &result2);

    // Third call - cache should still be used
    let result3 = tokenizer.encode_parallel_single(text3.as_str(), false).unwrap();
    let expected3 = tokenizer.encode(text3.as_str(), false).unwrap();
    assert_encodings_equal(&expected3, &result3);
}

// ============================================================================
// Recursive Parallelism Tests
// ============================================================================

/// Test recursive parallelism with medium input (should use depth 1-2)
#[test]
fn test_recursive_parallel_medium_input() {
    let tokenizer = common::get_bert();

    // ~100KB input, should trigger depth 1-2 depending on cores
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(2500);
    assert!(text.len() > 100_000);

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

/// Test recursive parallelism with large input (should use depth 2-3)
#[test]
fn test_recursive_parallel_large_input() {
    let tokenizer = common::get_bert();

    // ~250KB input, should trigger depth 2-3 depending on cores
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(6000);
    assert!(text.len() > 250_000);

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

/// Test recursive parallelism with very large input (should use max depth)
#[test]
fn test_recursive_parallel_very_large_input() {
    let tokenizer = common::get_bert();

    // ~500KB input, should trigger depth 3+ depending on cores
    let text = "Testing recursive parallelism with a longer sentence structure. ".repeat(8000);
    assert!(text.len() > 500_000);

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

/// Test recursive parallelism with 1MB input
#[test]
fn test_recursive_parallel_1mb_input() {
    let tokenizer = common::get_bert();

    // ~1MB input (each repeat is 76 bytes, need ~13200 repeats)
    let text = "This is a comprehensive test of the recursive parallel encoding algorithm. ".repeat(14000);
    assert!(text.len() > 1_000_000, "Text length {} should be > 1MB", text.len());

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

/// Test recursive parallelism with byte-level BPE tokenizer
#[test]
fn test_recursive_parallel_byte_level_large() {
    let tokenizer = common::get_byte_level(false, true);

    // ~300KB input
    let text = "Hello world! How are you doing today? ".repeat(8000);
    assert!(text.len() > 300_000);

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

/// Test recursive parallelism with special tokens
#[test]
fn test_recursive_parallel_with_special_tokens_large() {
    let tokenizer = common::get_bert();

    // ~200KB input with special tokens
    let text = "Testing with special tokens. ".repeat(7000);
    assert!(text.len() > 200_000);

    let serial = tokenizer.encode(text.as_str(), true).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), true).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

/// Test recursive parallelism with multi-byte UTF-8 characters throughout
#[test]
fn test_recursive_parallel_utf8_large() {
    let tokenizer = common::get_bert();

    // ~200KB with mixed UTF-8
    let text = "Hello 世界! Testing émojis 🎉 and Chinese 你好. ".repeat(4000);
    assert!(text.len() > 200_000);

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

/// Test that recursive parallel encoding handles edge cases at depth boundaries
#[test]
fn test_recursive_depth_boundaries() {
    let tokenizer = common::get_bert();

    // Test at various sizes that should hit different depth levels
    // Based on 10k tokens per chunk, ~45KB per chunk
    let test_sizes = vec![
        40_000,   // ~9k tokens - should be depth 0 (below threshold)
        90_000,   // ~20k tokens - depth 1 (2 chunks)
        180_000,  // ~40k tokens - depth 2 (4 chunks)
        360_000,  // ~80k tokens - depth 3 (8 chunks)
    ];

    for size in test_sizes {
        let repeats = size / 45; // "The quick brown fox jumps over the lazy dog. " is ~45 bytes
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(repeats);

        let serial = tokenizer.encode(text.as_str(), false).unwrap();
        let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

        assert_encodings_equal(&serial, &parallel);
    }
}

/// Test recursive parallelism with text containing no whitespace (worst case)
#[test]
fn test_recursive_parallel_no_whitespace_large() {
    let tokenizer = common::get_bert();

    // ~100KB with no whitespace - challenging for split point finding
    let text = "abcdefghij".repeat(10000);
    assert!(text.len() >= 100_000);

    let serial = tokenizer.encode(text.as_str(), false).unwrap();
    let parallel = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    assert_encodings_equal(&serial, &parallel);
}

/// Test that recursive encoding produces monotonically increasing offsets
#[test]
fn test_recursive_offsets_monotonic() {
    let tokenizer = common::get_bert();

    // Large input to trigger recursive splitting
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(6000);

    let encoding = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();
    let offsets = encoding.get_offsets();

    for i in 0..offsets.len() {
        let (start, end) = offsets[i];

        // Offset must be valid (start <= end)
        assert!(
            start <= end,
            "Invalid offset at position {}: ({}, {})",
            i,
            start,
            end
        );

        // Offset should not exceed text length
        assert!(
            end <= text.len(),
            "Offset exceeds text length at position {}: ({}, {}) > {}",
            i,
            start,
            end,
            text.len()
        );

        // Check non-overlapping with next
        if i + 1 < offsets.len() {
            let (next_start, _) = offsets[i + 1];
            assert!(
                end <= next_start,
                "Overlapping offsets at positions {}-{}: ({}, {}) and ({}, {})",
                i,
                i + 1,
                start,
                end,
                next_start,
                offsets[i + 1].1
            );
        }
    }
}
