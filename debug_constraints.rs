// Simple debug test
use std::collections::HashMap;

fn main() {
    println!("Debug test started");
    
    // Test map insertion
    let mut map = HashMap::new();
    map.insert("test".to_string(), 42);
    println!("Map after insert: {:?}", map);
}
