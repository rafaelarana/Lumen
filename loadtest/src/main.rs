//! Lumen load test driver.
//!
//! Uses Goose (Rust load-testing framework, async, Locust-inspired) to fire
//! concurrent search requests at the deployed Databricks App, head-to-head
//! across Standard and Turbo modes, measuring throughput and full latency
//! percentiles.
//!
//! ## Request mix
//!
//! Each iteration randomly picks:
//!   • path:   50% Standard (/api/search) · 50% Turbo (/api/search/fast)
//!   • mode:   70% semantic            · 30% hybrid
//!
//! ⇒ 4 named request buckets, tracked separately in the report:
//!   standard:semantic · standard:hybrid · turbo:semantic · turbo:hybrid
//!
//! Override the path mix with `--turbo-pct N` (0–100).
//!
//! ## Usage
//!
//!     LUMEN_TOKEN=$(databricks auth token --profile azure-video \
//!         | jq -r .access_token) \
//!     cargo run --release -- \
//!         --host https://lumen-recommender-7405604561430667.7.azure.databricksapps.com \
//!         -u 20 -r 5 -t 2m \
//!         --report-file report.html --no-reset-metrics
//!
//! Note on Turbo: the first ~minute of the test warms the in-process LRU
//! cache as the 100 sample queries cycle through. Steady-state hit rate
//! converges to ~100% once every query has been seen at least once.

use std::env;

use goose::prelude::*;
use rand::seq::SliceRandom;
use serde_json::json;

mod queries;
use queries::QUERIES;

/// Mode mix: 70% semantic, 30% hybrid.
const SEMANTIC_WEIGHT: u8 = 70;

/// Path mix: default 50% Standard, 50% Turbo. Override with --turbo-pct.
const DEFAULT_TURBO_WEIGHT: u8 = 50;

fn turbo_weight() -> u8 {
    // Allow `LUMEN_TURBO_PCT=N` to control the path split without recompiling.
    env::var("LUMEN_TURBO_PCT")
        .ok()
        .and_then(|s| s.parse::<u8>().ok())
        .map(|n| n.min(100))
        .unwrap_or(DEFAULT_TURBO_WEIGHT)
}

async fn search(user: &mut GooseUser) -> TransactionResult {
    let turbo_pct = turbo_weight();

    // Scope the (non-Send) ThreadRng so it's dropped before any `.await`.
    let (query, mode, path) = {
        let mut rng = rand::thread_rng();
        let q = QUERIES.choose(&mut rng).copied().unwrap_or("chair");
        let m = if rand::random::<u8>() % 100 < SEMANTIC_WEIGHT {
            "semantic"
        } else {
            "hybrid"
        };
        let p = if rand::random::<u8>() % 100 < turbo_pct {
            "turbo"
        } else {
            "standard"
        };
        (q, m, p)
    };

    // URL + display name resolved statically so we keep &'static str slices
    // (no allocation per request).
    let (url, name): (&'static str, &'static str) = match (path, mode) {
        ("turbo", "semantic")    => ("/api/search/fast", "turbo:semantic"),
        ("turbo", "hybrid")      => ("/api/search/fast", "turbo:hybrid"),
        ("standard", "semantic") => ("/api/search",      "standard:semantic"),
        ("standard", "hybrid")   => ("/api/search",      "standard:hybrid"),
        _                        => ("/api/search",      "standard:semantic"),
    };

    let body = json!({
        "q": query,
        "mode": mode,
        "limit": 20,
    });

    let token = env::var("LUMEN_TOKEN").unwrap_or_default();

    let request_builder = user
        .get_request_builder(&GooseMethod::Post, url)?
        .bearer_auth(&token)
        .json(&body);

    let goose_request = GooseRequest::builder()
        .name(name)
        .set_request_builder(request_builder)
        .build();

    let mut response = user.request(goose_request).await?;

    if let Ok(r) = response.response.as_ref() {
        let status = r.status();
        if !status.is_success() {
            return user
                .set_failure(
                    &format!("{name} returned HTTP {status}"),
                    &mut response.request,
                    None,
                    None,
                )
                .map(|_| ());
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), GooseError> {
    if env::var("LUMEN_TOKEN").is_err() {
        eprintln!(
            "warning: LUMEN_TOKEN is not set — requests will be unauthenticated.\n\
             Export it before running:\n  \
             export LUMEN_TOKEN=$(databricks auth token --profile azure-video | jq -r .access_token)"
        );
    }

    eprintln!(
        "config: turbo_pct={}% (set LUMEN_TURBO_PCT to override), semantic_pct={}%, {} queries",
        turbo_weight(),
        SEMANTIC_WEIGHT,
        QUERIES.len()
    );

    GooseAttack::initialize()?
        .register_scenario(
            scenario!("Search")
                .register_transaction(transaction!(search).set_name("search")),
        )
        .execute()
        .await?;

    Ok(())
}
