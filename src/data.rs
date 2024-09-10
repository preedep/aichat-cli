use log::debug;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Serialize, Deserialize)]
struct PIIDataDescription {
    #[serde(rename = "pii_description")]
    pii_descriptions: Vec<String>,
    #[serde(rename = "exclude_pii_description")]
    exclude_pii_descriptions: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MQTopicDescription {
    #[serde(rename = "business_module")]
    business_module: String,
    #[serde(rename = "topic_name")]
    topic_name: String,
    #[serde(rename = "publisher")]
    publisher: String,
    #[serde(rename = "remark")]
    remark: String,
}
#[derive(Debug, Serialize, Deserialize)]
struct MQDataDescription {
    #[serde(rename = "mq_data_background")]
    mq_descriptions: String,
    #[serde(rename = "mq_data_current_state")]
    mq_data_current_state: String,
    #[serde(rename = "mq_technology")]
    mq_technology: String,
    #[serde(rename = "mq_pub_sub_topics")]
    mq_pub_sub_topics: Vec<MQTopicDescription>,
}
// Function to load knowledge from a file (Refactor knowledge loading logic)
pub fn load_pii_knowledge(file_path: &str) -> String {
    let file_content = fs::read_to_string(file_path).expect("Failed to read JSON file");
    let parsed_json: PIIDataDescription =
        serde_json::from_str(&file_content).expect("Failed to parse JSON");

    debug!("Parsed JSON: {:?}", parsed_json);

    let mut knowledge = String::new();
    knowledge.push_str(
        "Here is the knowledge about Category of PII (Personal Identifiable Information) :\n",
    );
    for desc in parsed_json.pii_descriptions {
        knowledge.push_str(&desc);
        knowledge.push_str("\n");
    }
    knowledge.push_str(
        "Here is the knowledge about Category of Non-PII (Personal Identifiable Information) :\n",
    );
    for desc in parsed_json.exclude_pii_descriptions {
        knowledge.push_str(&desc);
        knowledge.push_str("\n");
    }
    knowledge
}
pub fn load_mq_knowledge(file_path: &str) -> String {
    let file_content = fs::read_to_string(file_path).expect("Failed to read JSON file");
    let parsed_json: MQDataDescription =
        serde_json::from_str(&file_content).expect("Failed to parse JSON");

    debug!("Parsed JSON: {:?}", parsed_json);

    let mut knowledge = String::new();
    knowledge.push_str("Here is the knowledge about Message sync MQ Pub/Sub :\n");
    knowledge.push_str(&parsed_json.mq_descriptions);
    knowledge.push_str("\n");
    knowledge.push_str("Here is the knowledge about Message sync MQ Pub/Sub Current State :\n");
    knowledge.push_str(&parsed_json.mq_data_current_state);
    knowledge.push_str("\n");
    knowledge.push_str("Here is the knowledge about Message sync MQ Pub/Sub Technology :\n");
    knowledge.push_str(&parsed_json.mq_technology);
    knowledge.push_str("\n");
    knowledge.push_str("Here is the knowledge about Message sync MQ Pub/Sub Topics :\n");
    for topic in parsed_json.mq_pub_sub_topics {
        knowledge.push_str("Business Module: ");
        knowledge.push_str(&topic.business_module);
        knowledge.push_str("\n");
        knowledge.push_str("Topic Name or Topic String: ");
        knowledge.push_str(&topic.topic_name);
        knowledge.push_str("\n");
        knowledge.push_str("Publisher: ");
        knowledge.push_str(&topic.publisher);
        knowledge.push_str("\n");
        knowledge.push_str("Remark: ");
        knowledge.push_str(&topic.remark);
        knowledge.push_str("\n");
    }
    knowledge.push_str("\n");
    knowledge
}