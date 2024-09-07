use langchain_rust::chain::{Chain, LLMChainBuilder};
use langchain_rust::language_models::llm::LLM;
use langchain_rust::llm::{AzureConfig, OpenAI};
use langchain_rust::prompt::HumanMessagePromptTemplate;
use langchain_rust::schemas::Message;
use langchain_rust::{fmt_message, fmt_template, message_formatter, prompt_args, template_fstring};
use log::{debug, error, info};
use std::io;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    pretty_env_logger::init();

    dotenv::dotenv().ok();

    let open_ai_url = std::env::var("OPEN_AI_SERVICE_URL").expect("OPEN_AI_SERVICE_URL is not set");
    let open_ai_key = std::env::var("OPEN_AI_SERVICE_KEY").expect("OPEN_AI_SERVICE_KEY is not set");

    debug!("open_ai_url: {}", open_ai_url);

    let azure_config = AzureConfig::default()
        .with_api_base(open_ai_url)
        .with_api_key(open_ai_key)
        .with_api_version("2023-03-15-preview")
        .with_deployment_id("gpt-4");

    let open_ai = OpenAI::new(azure_config);

    // Create an atomic flag to track whether Ctrl-C has been pressed
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    // Set up the Ctrl-C handler
    ctrlc::set_handler(move || {
        debug!("\nCtrl-C detected, exiting...");
        r.store(false, Ordering::SeqCst); // Set the flag to false
    })
    .expect("Error setting Ctrl-C handler");

    while running.load(Ordering::SeqCst) {
        // Check if Ctrl-C was pressed before continuing the loop
        if !running.load(Ordering::SeqCst) {
            break;
        }
        print!("Please enter some text and press Enter: ");
        // Flush stdout to ensure the message is displayed immediately
        io::stdout().flush().unwrap();
        let mut input = String::new();
        let result = io::stdin()
            .read_line(&mut input);

        // Check if Ctrl-C was pressed while waiting for input
        if !running.load(Ordering::SeqCst) {
            break;
        }
        if result.is_err(){
            error!("Error reading input: {:?}", result);
            continue;
        }

        // Trim the input and check if it's empty or "exit"
        let input = input.trim();
        if input.is_empty() || input == "exit" {
            break;
        }


        let resp = open_ai.invoke(input).await.unwrap();
        info!("{}", resp);

        // We can also guide it's response with a prompt template. Prompt templates are used to convert raw user input to a better input to the LLM.
        let prompt = message_formatter![
            fmt_message!(Message::new_system_message(
                "You are world class technical documentation writer."
            )),
            fmt_template!(HumanMessagePromptTemplate::new(template_fstring!(
                "{input}", "input"
            )))
        ];

        let chain = LLMChainBuilder::new()
            .prompt(prompt)
            .llm(open_ai.clone())
            .build()
            .unwrap();

        match chain
            .invoke(prompt_args! {
            "input" => "Quien es el escritor de 20000 millas de viaje submarino",
               })
            .await
        {
            Ok(result) => {
                info!("Result: {:?}", result);
            }
            Err(e) => panic!("Error invoking LLMChain: {:?}", e),
        }
    }

    Ok(())
}
