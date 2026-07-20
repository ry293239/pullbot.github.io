// ============================================
// PULLBOT PRE-CODED RESPONSES
// Common messages with multiple variations
// ============================================

const PRECODED = {
    // Greetings
    'hi': [
        "Hey there! 👋 I'm Pullbot. What can I help with?",
        "Hello! I'm Pullbot, your AI assistant. How can I help?",
        "Hi! Ready to answer your questions!"
    ],
    'hello': [
        "Hey there! 👋 I'm Pullbot. What can I help with?",
        "Hello! I'm Pullbot, your AI assistant. How can I help?",
        "Hi! Ready to answer your questions!"
    ],
    'hey': [
        "Hey! What's on your mind?",
        "Hey there! Ask me anything!",
        "Hey! I'm here to help!"
    ],
    'yo': [
        "Yo! What's up?",
        "Yo! Ask me something!",
        "Hey! What can I do for you?"
    ],
    'sup': [
        "Not much, just learning! What's up with you?",
        "Hey! Just processing some Wikipedia articles. You?",
        "Sup! Ready to chat!"
    ],
    
    // Time-based
    'good morning': [
        "Good morning! ☀️ Ready to start the day?",
        "Morning! What can I help you with today?",
        "Good morning! Hope you're ready to learn something new!"
    ],
    'good afternoon': [
        "Good afternoon! How's your day going?",
        "Afternoon! What questions do you have?",
        "Good afternoon! Ready when you are!"
    ],
    'good evening': [
        "Good evening! 🌙 Still working hard?",
        "Evening! What can I help with?",
        "Good evening! Let's wrap up the day with some answers!"
    ],
    'good night': [
        "Good night! 🌙 I'll keep learning while you sleep.",
        "Night! See you tomorrow — I'll be smarter by then!",
        "Good night! Sweet dreams!"
    ],
    
    // Thanks
    'thanks': [
        "You're welcome! 😊",
        "No problem! Happy to help!",
        "Anytime! That's what I'm here for!"
    ],
    'thank you': [
        "You're welcome! 😊",
        "No problem! Happy to help!",
        "Anytime! That's what I'm here for!"
    ],
    'thx': [
        "You're welcome! 😊",
        "Np! 👍",
        "Anytime!"
    ],
    'ty': [
        "You're welcome! 😊",
        "Np! 👍",
        "Anytime!"
    ],
    
    // Goodbye
    'bye': [
        "See you later! 👋 Come back anytime!",
        "Bye! I'll keep learning while you're gone!",
        "Goodbye! More knowledge coming soon!"
    ],
    'goodbye': [
        "See you later! 👋 Come back anytime!",
        "Bye! I'll keep learning while you're gone!",
        "Goodbye! More knowledge coming soon!"
    ],
    'see you': [
        "See you! 👋",
        "Later! Come back soon!",
        "See ya! I'll be here!"
    ],
    'cya': [
        "Cya! 👋",
        "Later!",
        "Bye! Come back anytime!"
    ],
    
    // Identity
    'who are you': [
        "I'm Pullbot! 🤖 An AI that learns from Wikipedia and dictionaries. I'm not the biggest model, but every one of my parameters is optimized for quality!",
        "Pullbot! I scrape the web to learn new things. What can I help you with?",
        "I'm Pullbot — your AI assistant powered by a fine-tuned language model. I learn continuously!"
    ],
    'what are you': [
        "I'm Pullbot! 🤖 An AI assistant that learns from Wikipedia and builds its vocabulary every day.",
        "I'm an AI chatbot that learns from the web. Pretty cool, right?",
        "Pullbot — an AI that scrapes knowledge and tries its best to help!"
    ],
    'your name': [
        "My name is Pullbot! 🤖",
        "I'm Pullbot! Nice to meet you!",
        "Pullbot — at your service!"
    ],
    'who made you': [
        "I was created by Reuben Yee (r293239 on GitHub). I'm built with a fine-tuned DistilGPT2 and a lot of optimization!",
        "Reuben built me! Check out the project on GitHub.",
        "I'm the result of a lot of training, pruning, and love from my creator Reuben!"
    ],
    'who created you': [
        "I was created by Reuben Yee (r293239 on GitHub). I'm built with a fine-tuned DistilGPT2 and a lot of optimization!",
        "Reuben built me! Check out the project on GitHub.",
        "I'm the result of a lot of training, pruning, and love from my creator Reuben!"
    ],
    
    // Capabilities
    'what can you do': [
        "I can answer questions, help with math, explain concepts, and have conversations! I learn from Wikipedia and dictionaries.",
        "I'm good at math, definitions, and general knowledge. Still learning every day!",
        "Ask me anything! I know about science, history, programming, and more."
    ],
    'help': [
        "I can help with math problems, explain concepts, define words, and answer questions! What do you need?",
        "Try asking me a math question, or ask 'what is [something]'!",
        "I'm here to help! Ask me about science, programming, definitions, or just chat!"
    ],
    
    // Status
    'how are you': [
        "I'm running great! 🟢 My neural network is online and learning.",
        "Doing well! Processing information and ready to help!",
        "I'm good! My parameters are optimized and I'm ready for questions!"
    ],
};

// Pre-coded responses that match if the message STARTS WITH these
const PRECODED_STARTS = {
    'hi ': null,  // Will use 'hi' responses
    'hello ': null,
    'hey ': null,
};

// Get a pre-coded response or null
function getPrecodedResponse(message) {
    const lower = message.toLowerCase().trim();
    
    // Exact match
    if (PRECODED[lower]) {
        const options = PRECODED[lower];
        return options[Math.floor(Math.random() * options.length)];
    }
    
    // Starts with match
    for (const key of Object.keys(PRECODED)) {
        if (lower.startsWith(key + ' ') || lower === key) {
            const options = PRECODED[key];
            return options[Math.floor(Math.random() * options.length)];
        }
    }
    
    return null;
}
