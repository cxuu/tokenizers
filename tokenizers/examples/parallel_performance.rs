use std::time::Instant;
use tokenizers::Tokenizer;

fn main() {
    // Create BERT tokenizer from pre-trained model
    let tokenizer = Tokenizer::from_file("data/bert-base-uncased.json")
        .or_else(|_| {
            // Fallback: try roberta which is also available
            Tokenizer::from_file("data/roberta.json")
        })
        .expect("Tokenizer file not found - run 'make test' first");

    // Test different sizes - go much larger to see the speedup
    let sizes = vec![
        10_000,     // 10KB
        50_000,     // 50KB
        100_000,    // 100KB
        500_000,    // 500KB
        1_000_000,  // 1MB - half million tokens
        2_000_000,  // 2MB
        5_000_000,  // 5MB
    ];

    println!("Benchmarking parallel vs serial encoding...\n");

    for size in sizes {
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(size / 45);
        let actual_size = text.len();
        let estimated_tokens = actual_size / 5; // ~5 bytes per token for this text

        // Warmup
        let _ = tokenizer.encode(text.as_str(), false).unwrap();
        let _ = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();

        // Serial (reduce iterations for very large inputs)
        let iters = if size > 2_000_000 { 1 } else if size > 500_000 { 2 } else { 3 };
        let start = Instant::now();
        for _ in 0..iters {
            let _ = tokenizer.encode(text.as_str(), false).unwrap();
        }
        let serial_time = start.elapsed() / iters as u32;

        // Parallel
        let start = Instant::now();
        for _ in 0..iters {
            let _ = tokenizer.encode_parallel_single(text.as_str(), false).unwrap();
        }
        let parallel_time = start.elapsed() / iters as u32;

        let speedup = serial_time.as_secs_f64() / parallel_time.as_secs_f64();

        println!("Size: {}KB (~{}K tokens)", actual_size / 1000, estimated_tokens / 1000);
        println!("  Serial:   {:?}", serial_time);
        println!("  Parallel: {:?}", parallel_time);
        if speedup >= 1.1 {
            println!("  Speedup:  {:.2}x ✓\n", speedup);
        } else if speedup >= 0.95 {
            println!("  Speedup:  {:.2}x (marginal/overhead)\n", speedup);
        } else {
            println!("  Speedup:  {:.2}x (slower - check if fallback occurred)\n", speedup);
        }
    }
}
