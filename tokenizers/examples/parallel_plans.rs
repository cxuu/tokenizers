//! Print the parallel encoding strategy of each tokenizer file given as argument.
//!
//! cargo run --release --example parallel_plans -- data/*.json data/parallel_corpus/*.json
use tokenizers::Tokenizer;

fn main() {
    for path in std::env::args().skip(1) {
        match Tokenizer::from_file(&path) {
            Ok(tokenizer) => {
                let time = || {
                    let start = std::time::Instant::now();
                    let plan = tokenizer.parallel_plan();
                    (plan, start.elapsed().as_secs_f64() * 1e3)
                };
                let (plan, first) = time();
                let (_, again) = time();
                println!("{plan:?}\t{first:.2}ms then {again:.2}ms\t{path}");
            }
            Err(e) => println!("error\t{path}\t{e}"),
        }
    }
}
