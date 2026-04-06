from flask import Flask, request, jsonify, render_template
from chatbot_engine import FAQChatbot
import os

app = Flask(__name__)

# Initialize the chatbot loading from the local directory
faqs_path = os.path.join(os.path.dirname(__file__), 'faqs.json')
bot = FAQChatbot(faqs_path=faqs_path)

@app.route('/')
def index():
    # Render the chat UI
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
        
    user_message = data['message']
    
    # Get response from the bot
    bot_response = bot.get_response(user_message)
    
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    # Start the server
    # Running on port 5001 to prevent conflicts with the 5000/8000 ones if any.
    app.run(host='127.0.0.1', port=5001, debug=True)
