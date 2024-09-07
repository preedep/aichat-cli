use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use langchain_rust::chain::{Chain, LLMChainBuilder};
use langchain_rust::llm::{AzureConfig, OpenAI};
use langchain_rust::prompt::HumanMessagePromptTemplate;
use langchain_rust::schemas::Message;
use langchain_rust::{
    fmt_message, fmt_placeholder, fmt_template, message_formatter, prompt_args, template_fstring,
};
use log::{debug, error};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use std::{fs, io, thread};

// Function to load knowledge from a file (Refactor knowledge loading logic)
fn load_knowledge(file_path: &str) -> String {
    fs::read_to_string(file_path).unwrap_or_else(|_| {
        error!("Failed to load knowledge file.");
        String::new()
    })
}

// Function to create the Azure OpenAI configuration (Refactor LLM setup)
fn create_openai() -> OpenAI<AzureConfig> {
    let open_ai_url = std::env::var("OPEN_AI_SERVICE_URL").expect("OPEN_AI_SERVICE_URL is not set");
    let open_ai_key = std::env::var("OPEN_AI_SERVICE_KEY").expect("OPEN_AI_SERVICE_KEY is not set");

    debug!("open_ai_url: {}", open_ai_url);

    let azure_config = AzureConfig::default()
        .with_api_base(open_ai_url)
        .with_api_key(open_ai_key)
        .with_api_version("2023-03-15-preview")
        .with_deployment_id("gpt-4");

    OpenAI::new(azure_config)
}

// Function to handle user input (Refactor input handling logic)
fn get_user_input(running: Arc<AtomicBool>) -> Option<String> {
    if !running.load(Ordering::SeqCst) {
        return None;
    }

    print!(
        "{}",
        "Please enter some text and press Enter: ".bright_green()
    );
    io::stdout().flush().unwrap();

    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_err() {
        error!("Error reading input.");
        return None;
    }

    let input = input.trim();
    if input.is_empty() || input == "exit" {
        return None;
    }

    Some(input.to_string())
}

// Function to create a spinner (Refactor spinner creation)
fn create_spinner(message: &str) -> ProgressBar {
    let spinner = ProgressBar::new_spinner();
    spinner.set_message(format!("{} {}", "💡".blue(), message));
    spinner.set_style(
        ProgressStyle::with_template("{spinner:.green} {msg}")
            .unwrap()
            .tick_strings(&["|", "/", "-", "\\", "|", "/", "-", "\\"]),
    );
    spinner.enable_steady_tick(Duration::from_millis(120));
    spinner
}

// Function to handle the LLM chain execution and processing (Refactor LLM logic)
async fn process_with_llm(
    input: &str,
    knowledge: &str,
    history_list: &mut Vec<Message>,
    open_ai: &OpenAI<AzureConfig>,
    running: Arc<AtomicBool>,
    fn_callback: Box<dyn Fn() + 'static>,
) -> Result<String, Box<dyn std::error::Error>> {
    let prompt = message_formatter![
        fmt_message!(Message::new_system_message(
            "You are a world-class technical documentation writer. Use the following knowledge to answer the user's query."
        )),
        fmt_message!(Message::new_system_message(format!("Knowledge:\n{}", knowledge))),
        fmt_placeholder!("history"),
        fmt_template!(HumanMessagePromptTemplate::new(template_fstring!("{input}", "input")))
    ];

    let chain = LLMChainBuilder::new()
        .prompt(prompt)
        .llm(open_ai.clone())
        .build()?;

    let res = chain
        .invoke(prompt_args! {
            "input" => input,
            "knowledge" => knowledge,
            "history" => history_list
        })
        .await;

    fn_callback();

    if let Ok(result) = res {
        history_list.push(Message::new_ai_message(&result));
        typewriter(&result, 100, running);
        Ok(result)
    } else {
        Err(Box::new(res.err().unwrap()))
    }
}

// Function to display typing effect (Already refactored)
fn typewriter(text: &str, delay_ms: u64, running: Arc<AtomicBool>) {
    for c in text.chars() {
        if !running.load(Ordering::SeqCst) {
            break;
        }
        print!("{}", c.to_string().yellow());
        io::stdout().flush().unwrap();
        thread::sleep(Duration::from_millis(delay_ms));
    }
    println!();
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    pretty_env_logger::init();
    dotenv::dotenv().ok();

    // Load knowledge from a file
    let knowledge = "";//load_knowledge("dataset/app_info.json");
    let open_ai = create_openai();

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    // Set up the Ctrl-C handler
    ctrlc::set_handler(move || {
        debug!("\nCtrl-C detected, exiting...");
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    let mut history_list = Vec::new();
    // Main loop for user input and processing
    while running.load(Ordering::SeqCst) {
        if let Some(input) = get_user_input(running.clone()) {
            if input == "clear" {
                history_list.clear();
                continue;
            }

            history_list.push(Message::new_human_message(&input));

            let spinner = create_spinner("Asking...");
            let res = process_with_llm(
                &input,
                &knowledge,
                &mut history_list,
                &open_ai,
                running.clone(),
                Box::new(move || {
                    spinner.finish_and_clear();
                }),
            )
            .await;
            //spinner.finish_and_clear();

            if let Err(e) = res {
                error!("Error invoking LLMChain: {:?}", e);
            }
        } else {
            break;
        }
    }

    Ok(())
}
