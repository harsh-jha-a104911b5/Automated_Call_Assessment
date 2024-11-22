import json
import os
import google.generativeai as genai

# Set your API key as an environment variable
os.environ["API_KEY"] = "AIzaSyC_YhKLA5vhtiqiRJTur93duTc4jDs79vg"

# Configure the Gemini model with your API key
genai.configure(api_key=os.environ["API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

def parse_conversation_data(data):
    """
    Parse the conversation data to organize it by speaker and extract emotions, sentiments, and text.
    """
    conversation_text = ""
    speaker_data = {
        "SPEAKER_00": {"emotions": [], "sentiments": [], "text": []},
        "SPEAKER_01": {"emotions": [], "sentiments": [], "text": []},
    }

    # Process conversation data
    for item in data:
        speaker = item['speaker']
        text = item['text']
        sentiment = item['sentiment']['label']
        emotion = item['emotion']

        conversation_text += f"{speaker}: {text}\n"

        if speaker in speaker_data:
            speaker_data[speaker]["emotions"].append(emotion)
            speaker_data[speaker]["sentiments"].append(sentiment)
            speaker_data[speaker]["text"].append(text)

    # Aggregate emotions and sentiments
    for speaker, details in speaker_data.items():
        details["emotions"] = ", ".join(set(details["emotions"])) or "No emotions detected."
        details["sentiments"] = ", ".join(set(details["sentiments"])) or "No sentiments detected."
        details["text"] = " ".join(details["text"])

    return conversation_text, speaker_data

def construct_prompt(conversation_text, speaker_data):
    """
    Construct a detailed prompt for Gemini based on the parsed conversation data,
    aimed at providing more detailed and accurate insights.
    """
    prompt = f"""
    You are an advanced conversational analysis model. Please analyze the following conversation and provide a detailed and structured summary, incorporating insights into the dynamics, emotions, and sentiments of the speakers.

    1. Main Topic(s):
       - Identify the primary topics discussed in the conversation. Summarize the overall subject matter and the specific themes explored.

    2. Dynamics of the Conversation:
       - Analyze the flow of the conversation. How do the speakers interact with each other? Are there shifts in tone, interruptions, or specific conversational patterns? Provide insights on the overall structure and progression of the dialogue.

    3. Speaker Analysis:
       - For each speaker, provide a detailed breakdown:
         - Key Points: Highlight the main ideas or contributions made by the speaker.
         - Predominant Emotions: Identify the primary emotions conveyed by the speaker, including any shifts or variations throughout the conversation.
         - Sentiments: Summarize the positive, negative, or neutral sentiments expressed by the speaker.
         - Behavior Analysis: What is the speaker's conversational style (e.g., assertive, passive, polite, confrontational)? Does the speaker express empathy, frustration, or any other notable behaviors?
         - Tone: Describe the tone used by the speaker (e.g., formal, casual, sarcastic, enthusiastic).
         
       - Example:
         - SPEAKER_00:
           Key Points: [Your answer here]
           Predominant Emotions: {speaker_data['SPEAKER_00']['emotions']}
           Sentiments: {speaker_data['SPEAKER_00']['sentiments']}
           Behavior Analysis: [Your answer here]
           Tone: [Your answer here]
         
         - SPEAKER_01:
           Key Points: [Your answer here]
           Predominant Emotions: {speaker_data['SPEAKER_01']['emotions']}
           Sentiments: {speaker_data['SPEAKER_01']['sentiments']}
           Behavior Analysis: [Your answer here]
           Tone: [Your answer here]

    4. Emotional Tone of the Conversation:
       - Identify the overall emotional tone of the conversation. Is the tone primarily positive, negative, neutral, or mixed? How do the emotions evolve throughout the discussion? Does the tone shift in response to specific events or statements?

    5. Key Insights and Learnings:
       - Provide a deeper analysis of the conversation. What can be learned from the conversation in terms of communication dynamics, conflict resolution, decision-making, etc.? Are there any hidden or implied meanings, non-verbal cues, or deeper emotional undertones that were conveyed but not explicitly stated?

    6. Actionable Suggestions:
       - Based on the analysis, what actionable suggestions can be provided? These could be related to improving communication, resolving conflict, or addressing underlying issues in the conversation. Provide specific recommendations that could enhance the conversation's outcome or achieve better results in future interactions.

    7. Conversation Context:
       - {conversation_text}
    """
    return prompt

def generate_summary(data):
    """
    Generate a structured summary based on the provided conversation data using the Gemini model.
    """
    conversation_text, speaker_data = parse_conversation_data(data)
    prompt = construct_prompt(conversation_text, speaker_data)

    # Generate the summary using the Gemini model
    try:
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error during model inference: {str(e)}"

def main():
    # Prompt the user for the JSON file path
    json_file_path = input("Enter the path to the JSON file from the previous analysis: ")

    try:
        with open(json_file_path, 'r') as file:
            conversation_data = json.load(file)
            print("Loaded conversation data:", conversation_data)  # Debug statement
            # Generate summary after loading the data
            summary = generate_summary(conversation_data)
            print("Generated Summary:\n", summary)

    except FileNotFoundError:
        print("File not found. Please provide a valid path.")
    except json.JSONDecodeError:
        print("Error decoding JSON. Please check the file format.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
