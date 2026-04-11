import os
import re
import uuid
import base64
import time
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from huggingface_hub import InferenceClient
from PIL import Image

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN_SD = os.getenv("HF_TOKEN_SD")
HF_TOKEN_SD_XL = os.getenv("HF_TOKEN_SD_XL")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
except Exception as e:
    print(f"Error configuring Gemini: {e}")
    gemini_model = None

try:
    client_sd = InferenceClient("stabilityai/stable-diffusion-xl-base-1.0", token=HF_TOKEN_SD)
    client_sd_xl = InferenceClient("stabilityai/stable-diffusion-xl-base-1.0", token=HF_TOKEN_SD_XL)
except Exception as e:
    print(f"Error configuring HuggingFace: {e}")
    client_sd = None
    client_sd_xl = None

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    return response

# ============ STORYTELLER CORE LOGIC (AGE-BASED) ============

def get_age_appropriate_story_instructions(age):
    try:
        age_int = int(age)
    except (ValueError, TypeError):
        age_int = 10
        
    if age_int < 7:
        return """
        Create a story for young children (under 7 years):
        - Use very simple words and short sentences
        - Include repetitive phrases and predictable patterns
        - Focus on basic emotions and clear moral lessons
        - Use lots of action words and sound effects
        - Keep the story fun, colorful, and easy to follow
        - Include friendly characters and happy endings
        - Vocabulary should be at preschool/kindergarten level
        - Make it engaging with simple adventures
        """
    elif age_int >= 7 and age_int < 13:
        return """
        Create a story for children aged 7–12 years:
        - Use age-appropriate vocabulary with some challenging words
        - Include more detailed descriptions and character development
        - Create a clear plot with beginning, middle, and end
        - Add some mild conflict or problem-solving elements
        - Include themes like friendship, courage, teamwork, or learning
        - Make characters relatable with realistic emotions
        - Keep the story engaging with some suspense or excitement
        - End with a satisfying resolution and positive message
        """
    elif age_int >= 13 and age_int <= 20:
        return """
        Create a story for teenagers aged 13–20 years:
        - Use sophisticated vocabulary and varied sentence structures
        - Develop complex characters with deeper motivations
        - Include nuanced themes like identity, relationships, or personal growth
        - Create compelling conflicts with realistic resolutions
        - Add layers of meaning and subtle symbolism
        - Balance action with emotional depth and introspection
        - Make the story thought-provoking while still entertaining
        - Address age-appropriate challenges and life lessons
        """
    else:
        return """
        Create a story for adults (20+ years):
        - Use advanced literary language and sophisticated prose
        - Develop multi-dimensional characters with psychological depth
        - Explore complex themes such as ambition, morality, love, loss, or existential questions
        - Create intricate plots with subtle foreshadowing and symbolism
        - Include moral ambiguity and nuanced perspectives
        - Use literary devices like metaphor, irony, and subtext
        - Balance narrative with philosophical or emotional resonance
        - Craft an ending that is meaningful and thought-provoking, not necessarily happy
        - Make the story intellectually and emotionally engaging
        """

def storyTeller(input_text, age=10, image_file=None):
    story_instructions = get_age_appropriate_story_instructions(age)
    if gemini_model is None:
        raise ValueError("Gemini model is not initialized.")
    
    prompt = f"""
{story_instructions}

USER TEXT CONTEXT: "{input_text}"

You are given the user's text prompt (if any) and an image of their drawing (if any).
1. Image Analysis:
   - If an image is provided and you can recognize what the drawing represents (e.g., a house, an animal, a car), identify it in a short phrase and output exactly: <drawing>the identified subject</drawing> (e.g., <drawing>a red car</drawing>).
   - If the image is just scribbles or indistinguishable, OR if no image is provided, output exactly: <drawing>scribbles</drawing>.
   
2. Story Generation:
   - If BOTH the user text and a recognizable drawing are present, combine them creatively (even if they conflict, e.g., a dinosaur text with spaceship drawing -> a dinosaur on a spaceship).
   - If ONLY a recognizable drawing is present (user text is empty), write the story entirely around the subject of the drawing.
   - If ONLY the user text is present (drawing is scribbles or empty), write the story entirely based on the user text.
   - If BOTH are empty/scribbles, write a creative, original story appropriate for the age group.

Tell the story in EXACTLY 4 paragraphs.
"""

    if image_file:
        response = gemini_model.generate_content([prompt, image_file])
    else:
        response = gemini_model.generate_content(prompt)

    raw_text = response.text
    
    drawing_desc = "scribbles"
    match = re.search(r"<drawing>(.*?)</drawing>", raw_text, re.IGNORECASE)
    if match:
        drawing_desc = match.group(1).strip()
        raw_text = re.sub(r"<drawing>.*?</drawing>", "", raw_text, flags=re.IGNORECASE).strip()

    # Calling ImageGen concurrently or blocking
    ImageGen(raw_text, drawing_desc)
    return raw_text, drawing_desc

# ============ QUIZBOT CORE LOGIC (AGE-BASED) ============

def get_age_appropriate_quiz_instructions(age):
    try:
        age_int = int(age)
    except (ValueError, TypeError):
        age_int = 10
        
    if age_int < 7:
        return """
        Age Group: Under 7 years (Young Children)
        - Use very simple vocabulary and short sentences
        - Focus on basic concepts like colors, numbers, characters, and main events
        - Keep each option extremely short (1–3 words)
        """
    elif age_int >= 7 and age_int < 13:
        return """
        Age Group: 7–12 years (Elementary to Middle School)
        - Use age-appropriate vocabulary with moderate complexity
        - Include questions about sequence of events, cause and effect, and character feelings
        - Keep each option short (1–4 words)
        """
    elif age_int >= 13 and age_int <= 20:
        return """
        Age Group: 13–20 years (Teenagers)
        - Use more sophisticated vocabulary and complex concepts
        - Focus on themes, character development, symbolism, and deeper meanings
        - Keep each option concise (2–5 words)
        """
    else:
        return """
        Age Group: 20+ years (Adults - Difficult Level)
        - Use advanced and sophisticated vocabulary
        - Create challenging questions that require deep analysis and critical thinking
        - Keep options brief (max 5–6 words)
        """

def parse_quiz_response(response):
    questions = re.split(r"(?=Question \d+:)", response.strip())
    parsed_questions = []

    for question in questions:
        lines = question.strip().split("\n")
        # Ensure we have enough valid lines before parsing
        valid_lines = [line.strip() for line in lines if line.strip()]
        if len(valid_lines) >= 6:
            question_text = re.sub(r"^Question \d+:\s*", "", valid_lines[0]).strip()
            options = valid_lines[1:5]
            correct_answer = valid_lines[5].replace("Correct Answer: ", "").strip()

            parsed_questions.append({
                "question": question_text,
                "options": options,
                "correctAnswer": correct_answer
            })
            
    return parsed_questions

def quizBot(input_text, age=10):
    age_instructions = get_age_appropriate_quiz_instructions(age)

    text = f"""
    Generate 10 multiple-choice questions based on the provided story. For each question, include:
    1. The question text.
    2. Four options (labeled a, b, c, d).
    3. skip 2 lines before starting the next question.

    {age_instructions}

    Use this exact format for the response:
    Question 1: [Your question here]
    a) [Option 1]
    b) [Option 2]
    c) [Option 3]
    d) [Option 4]
    Correct Answer: [Letter of the correct answer]

    Repeat for all 10 questions.
    """

    response = gemini_model.generate_content(
        f"in the following format: {text} \n Frame 10 questions and give 4 options with one correct answer on the following story: {input_text}"
    )

    quiz_data = parse_quiz_response(response.text)
    return quiz_data

# ============ IMAGE GENERATION ============

def ImageGen(text, drawing_desc="scribbles"):
    images_folder = "public/Images"
    public_folder = "public"

    os.makedirs(images_folder, exist_ok=True)
    
    # We remove the wiping of the folder to ensure concurrent requests do not conflict
    
    ParaList = [p for p in text.split("\n\n") if p.strip()]

    num_chunks = min(len(ParaList) - 1, 3) if len(ParaList) > 1 else 0

    for i in range(num_chunks):
        if not client_sd:
            print("HF Client SD is not initialized.")
            break
            
        try:
            if drawing_desc.lower() != "scribbles":
                SD_prompt = f"Generate a scenario-based prompt no more than 20 words to generate an image based on the following context: {ParaList[i]}. MUST visibly feature: {drawing_desc}"
            else:
                SD_prompt = f"Generate a scenario-based prompt no more than 20 words to generate an image based on the following context: {ParaList[i]}"
                
            PromptImage = gemini_model.generate_content(SD_prompt)
            image = client_sd.text_to_image(PromptImage.text)
            image.save(f"{images_folder}/Image{i + 1}.png")
        except Exception as e:
            print(f"Error generating Image{i+1}: {e}")

    try:
        if client_sd_xl and len(ParaList) > 0:
            last_para_idx = min(3, len(ParaList)-1)
            if drawing_desc.lower() != "scribbles":
                SD_prompt = f"Generate a scenario-based very short prompt to generate an image based on the following context: {ParaList[last_para_idx]}. MUST visibly feature: {drawing_desc}"
            else:
                SD_prompt = f"Generate a scenario-based very short prompt to generate an image based on the following context: {ParaList[last_para_idx]}"
                
            PromptImage = gemini_model.generate_content(SD_prompt)
            image = client_sd_xl.text_to_image(PromptImage.text)
            image.save(f"{public_folder}/Image4.png")
    except Exception as e:
        print(f"Error generating final image: {e}")


# ============ FLASK ROUTES ============

@app.route("/StoryTeller", methods=["POST"])
def story_teller_route():
    try:
        input_data = request.get_json()
        input_text = input_data.get("text", "")
        age = input_data.get("age", 10)
        image_base64 = input_data.get("image", None)
        
        image = None
        if image_base64:
            if image_base64.startswith("data:image"):
                image_base64 = image_base64.split(",")[1]
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))

        story_text, drawing_desc = storyTeller(input_text, age, image_file=image)
        return jsonify({"response": story_text, "drawing": drawing_desc})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/QuizBot", methods=["POST"])
def quiz_bot_route():
    try:
        input_data = request.get_json()
        input_text = input_data.get("text", "")
        age = input_data.get("age", 10)
        response = quizBot(input_text, age)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/LearnBot", methods=["POST"])
def learnBot():
    try:
        input_data = request.get_json()
        input_text = input_data.get("text", "")
        image_base64 = input_data.get("image", "")

        image = None
        if image_base64:
            if image_base64.startswith("data:image"):
                image_base64 = image_base64.split(",")[1]
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))

        if image and input_text:
            prompt = f"You are an Ai Agent; correct user's mistakes if wrong, respond kindly: {input_text}"
            response = gemini_model.generate_content([prompt, image])
        elif image:
            prompt = "You are an Ai Agent; correct user's text in the image if wrong, respond kindly."
            response = gemini_model.generate_content([prompt, image])
        elif input_text:
            prompt = f"You are an Ai Agent; correct user's text if wrong, respond kindly: {input_text}"
            response = gemini_model.generate_content(prompt)
        else:
            return jsonify({"error": "No valid input provided"}), 400

        return jsonify({"response": response.text})
    except Exception as e:
        print("Error in LearnBot:", e)
        return jsonify({"error": "Failed to generate response", "details": str(e)}), 500


@app.route("/AiSuggestionBot", methods=["GET"])
def aiSuggestionBot():
    try:
        text = "Language Development, Physical Development, Cognitive Skills, Communication Skills"
        response = gemini_model.generate_content(
            f"Give 4 brief suggestions for parents on how to improve their child's development focusing on: {text}"
        )
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/VideoAnalyzer", methods=["POST"])
def videoAnalyzer():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file part in the request"}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "No selected video file"}), 400

        # Safe unique temp path
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        unique_filename = f"{uuid.uuid4()}_{video_file.filename}"
        video_path = os.path.join(temp_dir, unique_filename)
        video_file.save(video_path)

        # Upload to Gemini
        gemini_video_file = genai.upload_file(path=video_path)
        
        while gemini_video_file.state.name == "PROCESSING":
            time.sleep(2)
            gemini_video_file = genai.get_file(gemini_video_file.name)

        if gemini_video_file.state.name == "FAILED":
            os.remove(video_path)
            return jsonify({"error": "Video processing failed on Gemini servers"}), 500

        prompt = "Pretend like you're talking to the person in the video. Analyze their body language or speech cautiously."
        response = gemini_model.generate_content([prompt, gemini_video_file], request_options={"timeout": 600})
        
        # Clean up local file after processing
        os.remove(video_path)

        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
