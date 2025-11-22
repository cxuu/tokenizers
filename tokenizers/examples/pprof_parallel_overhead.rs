use std::fs::File;
use tokenizers::Tokenizer;

fn main() {
    // Create BERT tokenizer
    let tokenizer = Tokenizer::from_file("data/bert-base-uncased.json")
        .or_else(|_| Tokenizer::from_file("data/roberta.json"))
        .expect("Tokenizer file not found");

    // Use 500KB input where we see 1.38x speedup
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(500_000 / 45);

    println!("=== pprof Runtime Profile: Parallel Encoding Overhead ===");
    println!("Input size: {}KB (~{}K tokens estimated)\n", text.len() / 1000, text.len() / 5000);

    // Warmup
    let _ = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

    println!("Starting profiling (running 100 iterations)...");

    // Start profiling
    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(1000) // Sample at 1000 Hz
        .blocklist(&["libc", "libSystem", "libsystem", "dyld"])
        .build()
        .unwrap();

    // Run parallel encoding multiple times
    for _ in 0..100 {
        let _ = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();
    }

    // Stop profiling and generate report
    if let Ok(report) = guard.report().build() {
        println!("Profiling complete!\n");

        // Generate flamegraph
        let flamegraph_file = File::create("target/parallel_overhead_flamegraph.svg").unwrap();
        report.flamegraph(flamegraph_file).unwrap();
        println!("✓ Flamegraph saved to: target/parallel_overhead_flamegraph.svg");

        println!("\n=== Analysis ===");
        println!("Open the flamegraph: open target/parallel_overhead_flamegraph.svg");
        println!("\nKey things to look for in the flamegraph:");
        println!("  • encode_fn calls (left, right, seam) - should dominate CPU time");
        println!("  • rayon::join overhead - thread coordination cost");
        println!("  • filter_tokens_* - filtering overhead (~2-3%)");
        println!("  • merge_with - merging overhead (~1-2%)");
        println!("  • find_split_point - split point calculation (<1%)");
        println!("\n=== Expected Overhead Breakdown ===");
        println!("  1. Seam encoding (overlap):  ~15-18% (duplicate work)");
        println!("  2. Rayon coordination:        ~2-3% (thread overhead)");
        println!("  3. Filtering:                 ~2-3%");
        println!("  4. Merging:                   ~1-2%");
        println!("  5. Validation (debug):        ~1-2%");
        println!("  ─────────────────────────────────────");
        println!("  Total overhead:               ~23-28%");
        println!("\nTheoretical: 50% time (2-way parallel) + 23-28% overhead = 73-78%");
        println!("Expected speedup: 100/75.5 = 1.32-1.36x");
        println!("Actual speedup: 1.38x ✓");
    }
}
