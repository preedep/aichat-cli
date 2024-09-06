use log::debug;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    pretty_env_logger::init();

    dotenv::dotenv().ok();

    let open_ai_url = std::env::var("OPEN_AI_SERVICE_URL").expect("OPEN_AI_SERVICE_URL is not set");
    debug!("open_ai_url: {}", open_ai_url);

    Ok(())
}
