use std::time::Instant;
use tokenizers::Tokenizer;

fn main() {
    // Enable debug output to see detailed timing
    std::env::set_var("DEBUG_PARALLEL", "1");

    // Create BERT tokenizer
    let tokenizer = Tokenizer::from_file("data/bert-base-uncased.json")
        .or_else(|_| Tokenizer::from_file("data/roberta.json"))
        .expect("Tokenizer file not found");

    // Use 500KB input where we see 1.38x speedup
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(500_000 / 45);

    println!("=== Manual Profiling: Parallel vs Serial Encoding ===");
    println!("Input size: {}KB (~{}K tokens estimated)\n", text.len() / 1000, text.len() / 5000);

    // Warmup
    let _ = tokenizer.encode(text.as_str(), false).unwrap();
    let _ = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    // Measure serial encoding (baseline)
    println!("--- Serial Encoding (Baseline) ---");
    let iterations = 20;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = tokenizer.encode(text.as_str(), false).unwrap();
    }
    let serial_total = start.elapsed();
    let serial_avg = serial_total / iterations;
    println!("Serial average: {:?}\n", serial_avg);

    // Measure parallel encoding with detailed breakdown
    println!("--- Parallel Encoding (Detailed) ---");
    let start = Instant::now();
    for i in 0..iterations {
        if i == 0 {
            // First iteration with debug output
            let _ = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();
            println!();
        } else {
            // Rest without debug to reduce noise
            std::env::remove_var("DEBUG_PARALLEL");
            let _ = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();
        }
    }
    let parallel_total = start.elapsed();
    let parallel_avg = parallel_total / iterations;

    println!("Parallel average: {:?}", parallel_avg);
    let speedup = serial_avg.as_secs_f64() / parallel_avg.as_secs_f64();
    println!("Speedup: {:.2}x\n", speedup);

    // Component breakdown analysis
    println!("=== Theoretical Overhead Analysis ===");
    let serial_ms = serial_avg.as_millis() as f64;
    let parallel_ms = parallel_avg.as_millis() as f64;

    println!("Serial time: {:.2}ms (100%)", serial_ms);
    println!("Parallel time: {:.2}ms ({:.1}% of serial)", parallel_ms, (parallel_ms / serial_ms) * 100.0);
    println!();

    // If we have 2-way parallelism, ideal would be 50% time
    let ideal_2way_ms = serial_ms / 2.0;
    let overhead_ms = parallel_ms - ideal_2way_ms;
    let overhead_pct = (overhead_ms / serial_ms) * 100.0;

    println!("Ideal 2-way parallel time: {:.2}ms (50%)", ideal_2way_ms);
    println!("Actual overhead: {:.2}ms ({:.1}% of serial)", overhead_ms, overhead_pct);
    println!();

    println!("=== Overhead Breakdown (Estimated) ===");
    println!("Based on the algorithm design:");
    println!("  1. Parallel encoding (left + right): ~50% of serial (50%)");
    println!("  2. Seam encoding (overlap region):   ~15-18% of serial");
    println!("  3. Filtering tokens:                 ~2-3% of serial");
    println!("  4. Merging encodings:                ~1-2% of serial");
    println!("  5. Validation (debug builds):        ~1-2% of serial");
    println!("  6. Synchronization overhead:         ~1-2% of serial");
    println!("  ───────────────────────────────────────────");
    println!("  Total expected:                      ~72-77% of serial");
    println!();
    println!("Expected speedup: 100 / 74.5 = 1.34x");
    println!("Actual speedup:   {:.2}x", speedup);
    println!();

    if speedup >= 1.30 && speedup <= 1.45 {
        println!("✓ Speedup matches expectations!");
    } else if speedup < 1.30 {
        println!("⚠ Speedup lower than expected - additional overhead present");
    } else {
        println!("✓ Speedup better than expected!");
    }
}

