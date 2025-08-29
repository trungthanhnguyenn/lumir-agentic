import gradio as gr
import os
import requests
import json
import uuid
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import logging

# Import time processing module
from module.pipeline import TimeProcessingPipeline
from logo_data import LOGO_BASE64

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
WINDMILL_API_URL = os.getenv("WINDMILL_API_URL", "http://localhost:5555")
WINDMILL_TOKEN = os.getenv("WINDMILL_TOKEN", "")
WINDMILL_ROUTE = "lumir_agentic_v2"


DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "vi")

# Initialize time processing pipeline
time_pipeline = TimeProcessingPipeline()

# Global variables for session management
chat_sessions: Dict[str, Dict[str, Any]] = {}

def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid.uuid4())

def suggest_birthday_from_session(session_id: str) -> str:
    """Generate a suggested birthday based on session ID hash"""
    # Use session ID hash to generate consistent birthday suggestion
    hash_obj = hashlib.md5(session_id.encode())
    hash_int = int(hash_obj.hexdigest()[:8], 16)
    
    # Generate day (1-28), month (1-12), year (1990-2005)
    day = (hash_int % 28) + 1
    month = ((hash_int >> 5) % 12) + 1
    year = 1990 + ((hash_int >> 9) % 16)
    
    return f"{day:02d}/{month:02d}/{year}"

# Validate birthday in DD/MM/YYYY format
def is_valid_birthday(date_str: str) -> bool:
    try:
        if not date_str:
            return False
        datetime.strptime(date_str.strip(), "%d/%m/%Y")
        return True
    except Exception:
        return False

def extract_current_day_from_query(user_query: str) -> str:
    """
    Extract Current_Day from user query using time processing module
    Returns date in "dd/mm/yyyy" format
    """
    try:
        # Process query through time pipeline
        time_result = time_pipeline.process_query(user_query)
        
        # Parse JSON response
        time_data = json.loads(time_result)
        
        # Extract calculated_date and convert to dd/mm/yyyy format
        if "calculated_date" in time_data and time_data["calculated_date"] != "unknown":
            # Convert from YYYY-MM-DD to dd/mm/yyyy
            date_obj = datetime.strptime(time_data["calculated_date"], "%Y-%m-%d")
            return date_obj.strftime("%d/%m/%Y")
        else:
            # If no specific date found, return today's date
            return datetime.now().strftime("%d/%m/%Y")
            
    except Exception as e:
        logger.warning(f"Failed to extract date from query: {e}")
        # Fallback to today's date
        return datetime.now().strftime("%d/%m/%Y")

def get_or_create_session(session_id: Optional[str] = None) -> Dict[str, Any]:
    """Get existing session or create new one"""
    if not session_id or session_id not in chat_sessions:
        new_session_id = generate_session_id()
        chat_sessions[new_session_id] = {
            "session_id": new_session_id,
            "history": [],
            "suggested_birthday": suggest_birthday_from_session(new_session_id),
            "created_at": datetime.now().isoformat()
        }
        return chat_sessions[new_session_id]
    return chat_sessions[session_id]

def call_windmill_api(
    question: str,
    session_id: str,
    history: List[Dict],
    birthday: str,
    name: str,
    username: str,
    language: str,
    current_day: str,
    filename: str,
    has_trading_data: bool
) -> str:
    """Call Windmill API with the provided parameters matching the router input fields"""
    
    # Prepare payload matching Windmill router input fields
    # Ensure empty strings are passed if user didn't provide values
    payload = {
        "Question": question if question else "",                    # Question* (string) - User Query
        "Session Id": session_id if session_id else "",             # Session Id* (string) - Required
        "History": history if history else [],                      # History (array) - Conversation history
        "Birthday": birthday if birthday else "",                   # Birthday* (string) - DD/MM/YYYY format
        "Language": language if language else "vi",                 # Language (string) - Output Language (vi/en)
        "Current Day": current_day if current_day else "",          # Current Day (string) - Date for numerology
        "Name": name if name else "",                               # Name* (string) - User Full Name
        "Username": username if username else "",                   # Username* (string) - In app username
        "Filename": filename if filename else "",                   # Filename (string) - Trading data file path
        "Has Trading Data": has_trading_data                       # Has Trading Data (boolean) - Flag for data presence
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {WINDMILL_TOKEN}"
    }
    
    try:
        logger.info(f"Calling Windmill API with session_id: {session_id}")
        logger.info(f"Raw inputs - question: '{question}', birthday: '{birthday}', name: '{name}', username: '{username}', language: '{language}'")
        logger.info(f"Processed payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            f"{WINDMILL_API_URL}/api/r/{WINDMILL_ROUTE}",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            # Some Windmill routes return a JSON-encoded string (e.g. "...\n...")
            # Parse it to preserve real newlines; fallback to unescaping common sequences
            try:
                parsed = json.loads(response.text)
                if isinstance(parsed, str):
                    return parsed
                # If it's an object with a 'text' field
                if isinstance(parsed, dict) and isinstance(parsed.get('text'), str):
                    return parsed['text']
                # Fallback: just return text with basic unescape for newlines/tabs/quotes
                return response.text.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
            except Exception:
                # If JSON parsing fails, try to unescape common sequences
                return response.text.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
        else:
            error_msg = f"API Error {response.status_code}: {response.text}"
            logger.error(error_msg)
            return error_msg
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Request failed: {str(e)}"
        logger.error(error_msg)
        return error_msg

def handle_file_upload(files, session_id_state):
    """Handle file upload and return status"""
    if not files:
        return "Không có file nào được upload.", session_id_state
    
    session = get_or_create_session(session_id_state)
    
    # For now, just return success message
    # In a real implementation, you would process and upload files
    uploaded_files = [f.name for f in files]
    status_msg = f"Đã upload thành công {len(uploaded_files)} file: {', '.join(uploaded_files)}"
    
    logger.info(f"Files uploaded for session {session['session_id']}: {uploaded_files}")
    return status_msg, session['session_id']

def chat_with_ai(
    message: str,
    history: List[Dict[str, str]],
    language: str,
    birthday: str,
    name: str,
    username: str,
    has_trading_data: bool,
    session_id_param: str
):
    """Main chat function with improved UX and non-intrusive birthday suggestion"""
    if not message.strip():
        # No message -> nothing to do
        yield history, "", session_id_param
        return
    
    # Get or create session
    session = get_or_create_session(session_id_param)
    session_id = session['session_id']
    
    # Use exact values from user input - no auto-filling
    birthday_to_use = birthday if birthday and str(birthday).strip() else ""
    
    # Determine current day for numerology using TimeProcessingPipeline
    current_day = extract_current_day_from_query(message)
    
    # Use exact filename from user input - no auto-filling
    filename = ""  # Let user provide if needed
    
    # 1. Immediately show user message in UI
    # Gradio Chatbot expects [user_message, bot_response] format
    history.append([message, None])
    yield history, "", session_id_param
    
    # 2. Show typing indicator as assistant placeholder
    if language == "Vietnamese":
        history.append([None, "💭 Tôi đang suy nghĩ để trả lời thích hợp..."])
    else:
        history.append([None, "💭 I'm thinking to answer your question appropriately..."])
    yield history, "", session_id_param
    
    # Build API history: take session history + current user
    api_history: List[Dict[str, str]] = session['history'].copy() if session['history'] else []
    api_history.append({"role": "user", "content": message})
    
    # Map language to Windmill format
    windmill_language = "vi" if language == "Vietnamese" else "en"
    
    # Call Windmill API with updated parameters
    response = call_windmill_api(
        question=message,
        session_id=session_id,
        history=api_history,
        birthday=birthday_to_use,
        name=name,
        username=username,
        language=windmill_language,
        current_day=current_day,
        filename=filename,
        has_trading_data=has_trading_data
    )
    
    # 3. Replace typing indicator with actual response (and append birthday hint if needed)
    final_response = response
    name_missing = (not name) or (not str(name).strip())
    birthday_missing_or_invalid = (not birthday) or (not str(birthday).strip()) or (not is_valid_birthday(str(birthday).strip()))
    if name_missing or birthday_missing_or_invalid:
        if language == "Vietnamese":
            final_response = f"{response}\n\n💡 Để nhận được câu trả lời cá nhân hoá hơn, vui lòng tạo tài khoản và đăng nhập vào hệ thống."
        else:
            final_response = f"{response}\n\n💡 To get more personalized answers, please sign up and login to the system."
    
    # Update the typing indicator with the final response (whether modified or original)
    # Gradio format: [user_message, bot_response]
    history[-1] = [None, final_response]
    
    # Add response to API history and persist (keep role-based format for API)
    api_history.append({"role": "assistant", "content": final_response})
    session['history'] = api_history
    
    yield history, "", session_id_param

def reset_chat(session_id_state):
    """Reset chat and create new session"""
    new_session = get_or_create_session()
    # Add welcome message to new session
    # Gradio format: [user_message, bot_response]
    welcome_msg = [None, "👋 Xin chào! Tôi là Lumir, trợ lý tài chính chuyên về trading.\n\n🔹 Tôi có thể giúp bạn:\n• Phân tích thị trường và xu hướng\n• Tư vấn chiến lược đầu tư\n• Giải đáp các câu hỏi về tài chính\n\nHãy cho tôi biết bạn cần hỗ trợ gì nhé! 😊"]
    new_session['history'] = [{"role": "assistant", "content": welcome_msg[1]}]  # Keep role-based for API
    return [welcome_msg], new_session['session_id'], "", "", "", has_trading_data_checkbox.value

def load_session_history(session_id_state):
    """Load chat history from session and convert to Gradio format"""
    if not session_id_state or session_id_state not in chat_sessions:
        return []
    session = chat_sessions[session_id_state]
    api_history = session.get('history', [])
    
    # Convert role-based history to Gradio format: [user_message, bot_response]
    gradio_history = []
    for turn in api_history:
        if turn.get("role") == "user":
            gradio_history.append([turn.get("content", ""), None])
        elif turn.get("role") == "assistant":
            if gradio_history and gradio_history[-1][1] is None:
                gradio_history[-1][1] = turn.get("content", "")
            else:
                gradio_history.append([None, turn.get("content", "")])
    
    return gradio_history

# CSS styling
CSS = """
.container {
    max-width: 1200px;
    margin: 0 auto;
}

/* Color palette based on logo */
:root {
    --primary-gold: #A6802D;
    --primary-gold-light: #C4A062;
    --primary-gold-dark: #8B6919;
    --secondary-black: #000000;
    --accent-gold: #D4AF37;
    --text-light: #FFFFFF;
    --text-dark: #FFFFFF;
    --bg-light: #000000;
    --bg-dark: #1A1A1A;
    --border-light: #333333;
    --shadow-light: rgba(166, 128, 45, 0.1);
    --shadow-medium: rgba(166, 128, 45, 0.2);
    --shadow-dark: rgba(166, 128, 45, 0.3);
}

.header {
    text-align: center;
    padding: 25px;
    background: linear-gradient(135deg, var(--primary-gold) 0%, var(--primary-gold-dark) 100%);
    color: var(--text-light);
    border-radius: 15px;
    margin-bottom: 25px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 25px;
    min-height: 90px;
    box-shadow: 0 8px 32px var(--shadow-medium);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.header-content {
    display: flex;
    align-items: center;
    gap: 25px;
    flex-wrap: wrap;
    justify-content: center;
}

.company-logo {
    width: auto;
    height: auto;
    max-width: 85px;
    max-height: 85px;
    min-width: 45px;
    min-height: 45px;
    border-radius: 12px;
    object-fit: contain;
    background: rgba(255, 255, 255, 0.15);
    padding: 10px;
    box-shadow: 0 6px 16px var(--shadow-medium);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    border: 2px solid rgba(255, 255, 255, 0.2);
}

.company-logo:hover {
    transform: scale(1.08) rotate(2deg);
    box-shadow: 0 12px 24px var(--shadow-dark);
    border-color: var(--accent-gold);
}

.header-text {
    text-align: left;
}

.header-text h1 {
    margin: 0;
    font-size: 2em;
    font-weight: 700;
    color: var(--text-light);
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
}

.header-text p {
    margin: 8px 0 0 0;
    font-size: 1em;
    font-weight: 500;
    color: var(--text-light);
    opacity: 0.9;
}

.chat-container {
    height: 500px;
    background: var(--bg-light);
    border-radius: 12px;
    border: 1px solid var(--border-light);
}

.config-panel {
    background: linear-gradient(135deg, var(--primary-gold) 0%, var(--primary-gold-dark) 100%);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px var(--shadow-light);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.typing-indicator {
    animation: pulse 1.5s ease-in-out infinite;
    color: var(--primary-gold);
    font-style: italic;
    font-weight: 500;
}

@keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
}

.message-input {
    border-radius: 25px !important;
    border: 2px solid var(--border-light) !important;
    padding: 15px 20px !important;
    background: var(--bg-light) !important;
    transition: all 0.3s ease !important;
}

.message-input:focus {
    border-color: var(--primary-gold) !important;
    box-shadow: 0 0 0 4px var(--shadow-light) !important;
    background: var(--bg-dark) !important;
    color: var(--text-light) !important;
}

.send-button {
    border-radius: 25px !important;
    background: linear-gradient(135deg, var(--primary-gold) 0%, var(--primary-gold-dark) 100%) !important;
    border: none !important;
    padding: 15px 30px !important;
    font-weight: 600 !important;
    color: var(--text-light) !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px var(--shadow-medium) !important;
}

.send-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px var(--shadow-dark) !important;
    background: linear-gradient(135deg, var(--primary-gold-light) 0%, var(--primary-gold) 100%) !important;
}

/* Custom styling for Gradio components */
.gradio-container {
    background: var(--bg-light) !important;
}

.gradio-container .main {
    background: transparent !important;
}

/* Body background */
body {
    background: var(--bg-light) !important;
}

/* Main app background */
#root, .gradio-container {
    background: var(--bg-light) !important;
}

/* Chat interface styling */
.chat-container {
    background: var(--bg-dark) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border-light) !important;
}

/* Sidebar styling */
.config-panel {
    background: var(--bg-dark) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border-light) !important;
}

/* Chat messages styling */
.chat-container .message {
    color: var(--text-light) !important;
}

.chat-container .user-message {
    background: var(--primary-gold) !important;
    color: var(--text-light) !important;
}

.chat-container .assistant-message {
    background: var(--bg-dark) !important;
    color: var(--text-light) !important;
}

/* Button styling */
button[data-testid="secondary"] {
    background: var(--bg-dark) !important;
    color: var(--text-light) !important;
    border: 1px solid var(--primary-gold) !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

button[data-testid="secondary"]:hover {
    background: var(--primary-gold) !important;
    color: var(--text-light) !important;
    transform: translateY(-1px) !important;
}

/* Dropdown styling */
select, .gr-dropdown {
    border: 2px solid var(--border-light) !important;
    border-radius: 8px !important;
    background: var(--bg-dark) !important;
    color: var(--text-light) !important;
}

select:focus, .gr-dropdown:focus {
    border-color: var(--primary-gold) !important;
    box-shadow: 0 0 0 3px var(--shadow-light) !important;
}

/* Checkbox styling */
input[type="checkbox"] {
    accent-color: var(--primary-gold) !important;
}

/* Textbox styling */
input[type="text"], textarea {
    border: 2px solid var(--border-light) !important;
    border-radius: 8px !important;
    background: var(--bg-dark) !important;
    color: var(--text-light) !important;
}

input[type="text"]:focus, textarea:focus {
    border-color: var(--primary-gold) !important;
    box-shadow: 0 0 0 3px var(--shadow-light) !important;
    background: var(--bg-dark) !important;
}

/* Placeholder text styling */
input[type="text"]::placeholder, textarea::placeholder {
    color: #888888 !important;
}

footer {
    visibility: hidden;
}

/* Responsive design for smaller screens */
@media (max-width: 768px) {
    .header {
        padding: 20px;
        gap: 20px;
        min-height: 70px;
    }
    .header-content {
        gap: 20px;
    }
    .company-logo {
        max-width: 70px;
        max-height: 70px;
        min-width: 35px;
        min-height: 35px;
        padding: 8px;
    }
    .header-text h1 {
        font-size: 1.6em;
    }
    .header-text p {
        font-size: 0.9em;
    }
}

@media (max-width: 480px) {
    .header {
        padding: 15px;
        gap: 15px;
        min-height: 60px;
    }
    .header-content {
        gap: 15px;
        flex-direction: column;
        text-align: center;
    }
    .company-logo {
        max-width: 60px;
        max-height: 60px;
        min-width: 30px;
        min-height: 30px;
        padding: 6px;
    }
    .header-text {
        text-align: center;
    }
    .header-text h1 {
        font-size: 1.4em;
    }
    .header-text p {
        font-size: 0.85em;
    }
}
"""

# Create custom theme
custom_theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#FEF7E6",
        c100="#FDEBC3",
        c200="#FBDC9B",
        c300="#F8CC72",
        c400="#F6BD49",
        c500="#A6802D",  # Primary gold
        c600="#8B6919",
        c700="#6F5214",
        c800="#533C0F",
        c900="#37260A",
        c950="#1B1305",
    ),
    secondary_hue=gr.themes.Color(
        c50="#F5F5F5",
        c100="#E7E7E7",
        c200="#D1D1D1",
        c300="#B0B0B0",
        c400="#888888",
        c500="#6D6D6D",
        c600="#5D5D5D",
        c700="#4F4F4F",
        c800="#454545",
        c900="#1A1A1A",  # Secondary black
        c950="#0A0A0A",
    ),
    neutral_hue=gr.themes.Color(
        c50="#FAFAFA",
        c100="#F5F5F5",
        c200="#E5E5E5",
        c300="#D4D4D4",
        c400="#A3A3A3",
        c500="#737373",
        c600="#525252",
        c700="#404040",
        c800="#262626",
        c900="#171717",
        c950="#0A0A0A",
    ),
    spacing_size=gr.themes.sizes.spacing_md,
    radius_size=gr.themes.sizes.radius_md,
    text_size=gr.themes.sizes.text_md,
)

# Create Gradio interface
with gr.Blocks(theme=custom_theme, css=CSS, title="LUMIR") as demo:
    
    # Header
    gr.HTML(f"""
    <div class="header">
        <div class="header-content">
            <img src="{LOGO_BASE64}" alt="BEQ-HOLDING Logo" class="company-logo">
            <div class="header-text">
                <h1>Lumir Smart Trading Platform</h1>
                <p>Powered by BEQ-HOLDINGS</p>
            </div>
        </div>
    </div>
    """)
    
    # Session state
    session_id_state = gr.State("")
    
    with gr.Row():
        # Left column - Configuration
        with gr.Column(scale=1):
            # gr.HTML('<div class="config-panel"><h3>⚙️ Configuration</h3></div>')
            
            # Language selection
            language_dropdown = gr.Dropdown(
                choices=["Vietnamese", "English"],
                value="Vietnamese",
                label="🌐 Language",
                info="Select response language (Vietnamese → vi, English → en)"
            )
            
            # Upload settings
            upload_checkbox = gr.Checkbox(
                label="📁 Enable Upload Mode",
                value=False,
                info="Enable file upload processing"
            )
            
            # File upload
            file_upload = gr.File(
                label="📎 Upload Files",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".txt", ".md"],
                visible=False
            )
            
            upload_status = gr.Textbox(
                label="Trạng thái Upload",
                interactive=False,
                visible=False
            )
            
            # Name input
            name_input = gr.Textbox(
                label="👤 Name*",
                placeholder="Your full name",
                info="Your full name for personalized responses"
            )
            
            # Username input
            username_input = gr.Textbox(
                label="🏷️ Username*",
                placeholder="Your username in our app",
                info="In app username"
            )
            
            # Birthday input
            birthday_input = gr.Textbox(
                label="🎂 Birthday* (DD/MM/YYYY)",
                placeholder="Your birthday",
                info="Used for personalized responses"
            )
            
            # Filename input
            filename_input = gr.Textbox(
                label="📁 Filename",
                placeholder="Your trading data file path",
                info="Path to trading data file"
            )
            
            # Has trading data checkbox
            has_trading_data_checkbox = gr.Checkbox(
                label="📊 Has Trading Data",
                value=False,
                info="Check if you have trading data to analyze"
            )
            
            # Advanced settings
            with gr.Accordion("🔧 Advanced Settings", open=False):
                rerank_checkbox = gr.Checkbox(
                    label="🔄 Enable Reranking",
                    value=False,
                    info="Improve search result relevance"
                )
                
                session_info = gr.Textbox(
                    label="Session ID",
                    interactive=False,
                    info="Current session ID"
                )
        
        # Right column - Chat interface
        with gr.Column(scale=2):
            # Chat interface
            chatbot = gr.Chatbot(
                label="💬 Chat",
                height=500,
                show_copy_button=True
            )
            
            # Message input
            with gr.Row():
                msg_input = gr.Textbox(
                    label="",
                    placeholder="Enter your message...",
                    scale=4,
                    lines=2,
                    elem_classes=["message-input"]
                )
                send_btn = gr.Button("Send 📤", scale=1, variant="primary", elem_classes=["send-button"])
            
            # Control buttons
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                new_session_btn = gr.Button("🆕 New Session", variant="secondary")
    
    # Has trading data checkbox - update state when changed
    def update_trading_data_state(checkbox_value):
        return checkbox_value
    
    has_trading_data_checkbox.change(
        fn=update_trading_data_state,
        inputs=[has_trading_data_checkbox],
        outputs=[has_trading_data_checkbox]
    )
    def toggle_upload_visibility(upload_enabled):
        return {
            file_upload: gr.update(visible=upload_enabled),
            upload_status: gr.update(visible=upload_enabled)
        }
    
    def initialize_session():
        session = get_or_create_session()
        # Add welcome message to new sessions
        if not session['history']:
            # Gradio format: [user_message, bot_response]
            welcome_msg = [None, "👋 Xin chào! Tôi là Lumir, trợ lý tài chính chuyên về trading.\n\n🔹 Tôi có thể giúp bạn:\n• Phân tích thị trường và xu hướng\n• Tư vấn chiến lược đầu tư\n• Trả lời các câu hỏi liên quan đến hệ thống LUMIR và LUMIR-AI\n• Giải đáp các câu hỏi về tài chính\n\nHãy cho tôi biết bạn cần hỗ trợ gì nhé! 😊"]
            session['history'] = [{"role": "assistant", "content": welcome_msg[1]}]  # Keep role-based for API
            initial_history = [welcome_msg]
        else:
            # Convert role-based history to Gradio format
            initial_history = load_session_history(session['session_id'])

        # Keep the current checkbox value; ensure empty strings for text inputs
        return session['session_id'], "", "", "", "", has_trading_data_checkbox.value, initial_history
    
    def clear_chat_fields(has_trading_data_state):
        """Clear chat fields but preserve checkbox state"""
        return "", "", "", "", has_trading_data_state
    
    def reset_chat_with_checkbox(has_trading_data):
        """Reset chat but preserve checkbox state"""
        new_session = get_or_create_session()
        # Add welcome message to new session
        # Gradio format: [user_message, bot_response]
        welcome_msg = [None, "👋 Xin chào! Tôi là Lumir, trợ lý tài chính chuyên về trading.\n\n🔹 Tôi có thể giúp bạn:\n• Phân tích thị trường và xu hướng\n• Tư vấn chiến lược đầu tư\n• Trả lời các câu hỏi liên quan đến hệ thống LUMIR và LUMIR-AI\n• Giải đáp các câu hỏi về tài chính\n\nHãy cho tôi biết bạn cần hỗ trợ gì nhé! 😊"]
        new_session['history'] = [{"role": "assistant", "content": welcome_msg[1]}]  # Keep role-based for API
        return [welcome_msg], new_session['session_id'], "", "", "", has_trading_data
    
    # Initialize session on load
    demo.load(
        fn=initialize_session,
        outputs=[session_id_state, birthday_input, name_input, username_input, filename_input, has_trading_data_checkbox, chatbot]
    )
    
    # Update session info display and load history when session changes
    # Note: Gradio State objects don't have change events, so we'll handle this differently
    
    # Toggle upload visibility
    upload_checkbox.change(
        fn=toggle_upload_visibility,
        inputs=[upload_checkbox],
        outputs=[file_upload, upload_status]
    )
    
    # File upload handler
    file_upload.upload(
        fn=handle_file_upload,
        inputs=[file_upload, session_id_state],
        outputs=[upload_status, session_id_state]
    )
    
    # Chat handlers with streaming support
    def submit_message(message, history, language, birthday, name, username, filename, has_trading_data, session_id):
        # Generator function for streaming
        for result in chat_with_ai(message, history, language, birthday, name, username, has_trading_data, session_id):
            yield result
    
    # Send message on button click
    send_btn.click(
        fn=submit_message,
        inputs=[
            msg_input, chatbot, language_dropdown, 
            birthday_input, name_input, username_input, filename_input, has_trading_data_checkbox, session_id_state
        ],
        outputs=[chatbot, msg_input, session_id_state]
    )
    
    # Send message on Enter
    msg_input.submit(
        fn=submit_message,
        inputs=[
            msg_input, chatbot, language_dropdown,
            birthday_input, name_input, username_input, filename_input, has_trading_data_checkbox, session_id_state
        ],
        outputs=[chatbot, msg_input, session_id_state]
    )
    
    # Clear chat
    clear_btn.click(
        fn=lambda: ([], ""),
        outputs=[chatbot, msg_input]
    ).then(
        fn=clear_chat_fields,
        inputs=[has_trading_data_checkbox],
        outputs=[birthday_input, name_input, username_input, filename_input, has_trading_data_checkbox]
    )
    
    # New session
    new_session_btn.click(
        fn=reset_chat_with_checkbox,
        inputs=[has_trading_data_checkbox],
        outputs=[chatbot, session_id_state, birthday_input, name_input, username_input, filename_input, has_trading_data_checkbox]
    )

if __name__ == "__main__":
    print("🚀 Starting Windmill AI Chat Interface...")
    print(f"📡 API URL: {WINDMILL_API_URL}")
    print(f"🔑 Token configured: {'Yes' if WINDMILL_TOKEN else 'No'}")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,  # Changed port to avoid conflict
        share=True,
        show_error=True
    )
