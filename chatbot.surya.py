import os
import random
import json
import nltk
import ssl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import datetime
import pickle

ssl._create_default_https_context = ssl._create_unverified_context


nltk.download("punkt")
nltk.download("wordnet")


lemmatizer = nltk.WordNetLemmatizer()

# Intents defined within the code
intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "How are you?", "Good morning", "Good evening"],
            "responses": ["Hello!", "Hi there!", "Hi, how can I assist you?"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "Goodbye", "See you later", "I am leaving"],
            "responses": ["Goodbye!", "See you later!", "Have a great day!"]
        },
        {
            "tag": "thanks",
            "patterns": ["Thanks", "Thank you", "I appreciate it"],
            "responses": ["You're welcome!", "No problem!", "Happy to help!"]
        },
        {
            "tag": "help",
            "patterns": ["Can you help me?", "I need assistance", "Help me"],
            "responses": ["Sure, what do you need help with?", "I'm here to assist!"]
        },
        {
        "tag": "about",
        "patterns": [
            "What can you do",
            "Who are you",
            "What are you",
            "What is your purpose"
        ],
        "responses": [
            "I am a chatbot",
            "My purpose is to assist you",
            "I can answer questions and provide assistance"
        ]
        },
        {
        "tag": "age",
        "patterns": [
            "How old are you",
            "What's your age"
        ],
        "responses": [
            "I don't have an age. I'm a chatbot.",
            "I was just born in the digital world.",
            "Age is just a number for me."
        ]
    },{
        "tag": "weather",
        "patterns": [
            "What's the weather like",
            "How's the weather today"
        ],
        "responses": [
            "I'm sorry, I cannot provide real-time weather information.",
            "You can check the weather on a weather app or website."
        ]
    },
    {
        "tag": "budget",
        "patterns": [
            "How can I make a budget",
            "What's a good budgeting strategy",
            "How do I create a budget"
        ],
        "responses": [
            "To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.",
            "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.",
            "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."
        ]
    },{
        "tag": "credit_score",
        "patterns": [
            "What is a credit score",
            "How do I check my credit score",
            "How can I improve my credit score"
        ],
        "responses": [
            "A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.",
            "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."
        ]
    },
    {
        "tag": "name",
        "patterns": [
            "What's your name",
            "Do you have a name",
            "What should I call you"
        ],
        "responses": [
            "You can call me Chatbot.",
            "My name is Chatbot."
        ]
    },{
        "tag": "favorite_color",
        "patterns": [
            "What's your favorite color",
            "Do you like any color"
        ],
        "responses": [
            "I'm a chatbot, so I don't have a favorite color.",
            "As an AI, I don't have preferences like humans do."
        ]
    },
    {
        "tag": "hobby",
        "patterns": [
            "What do you do for fun",
            "Do you have any hobbies"
        ],
        "responses": [
            "I don't have hobbies as I'm here to assist you.",
            "I'm always ready to chat with you, so that's my favorite thing to do!"
        ]
    },{
        "tag": "time",
        "patterns": [
            "What's the time",
            "Can you tell me the time"
        ],
        "responses": [
            "I'm sorry, but I don't have access to real-time information like the current time.",
            "You can check the time on your device or use a clock."
        ]
    },
    {
        "tag": "joke",
        "patterns": [
            "Tell me a joke",
            "Do you know any jokes",
            "Make me laugh"
        ],
        "responses": [
            "Why don’t scientists trust atoms? Because they make up everything!",
            "Why don't skeletons fight each other? They don't have the guts!",
            "Did you hear about the mathematician who’s afraid of negative numbers? He’ll stop at nothing to avoid them!"
        ]
    },{
        "tag": "food",
        "patterns": [
            "What's your favorite food",
            "Do you like to eat",
            "Tell me about food"
        ],
        "responses": [
            "As a chatbot, I don't eat, but I'm here to help with any questions you have about food!",
            "I may not have taste buds, but I can assist you in finding recipes or restaurants!"
        ]
    },
    {
        "tag": "movies",
        "patterns": [
            "What's your favorite movie",
            "Recommend me a movie",
            "Tell me about movies"
        ],
        "responses": [
            "As an AI, I don't watch movies, but I can suggest some popular ones like The Shawshank Redemption, Inception, or The Godfather.",
            "I may not have personal preferences, but I can help you find movies based on your taste!"
        ]
    },{
        "tag": "technology",
        "patterns": [
            "What's the latest tech news",
            "Tell me about technology",
            "What's new in tech"
        ],
        "responses": [
            "As an AI language model, I don't have real-time data, but you can check technology news websites for the latest updates.",
            "Technology is constantly evolving. Stay updated with tech blogs and news sites to know about the latest advancements!"
        ]
    },
    {
        "tag": "compliment",
        "patterns": [
            "You're awesome",
            "You're great",
            "I like you"
        ],
        "responses": [
            "Thank you! I'm just a program, but I'm here to assist and make your experience better.",
            "I'm glad you think so! My purpose is to help and provide useful information."
        ]
    },{
        "tag": "meaning_of_life",
        "patterns": [
            "What is the meaning of life",
            "Why are we here",
            "What's the purpose of life"
        ],
        "responses": [
            "The meaning of life is a philosophical question. Different people and cultures have different beliefs about it.",
            "The purpose of life is subjective and can vary from person to person. Some find purpose in relationships, careers, or helping others."
        ]
    },
    {
        "tag": "sports",
        "patterns": [
            "Tell me about sports",
            "What's your favorite sport",
            "Any sports news"
        ],
        "responses": [
            "While I don't have personal preferences, sports can be exciting! Keep an eye on sports news websites for the latest updates and scores.",
            "Sports are a great way to stay active and entertained. There's a wide range of sports to explore and enjoy!"
        ]
    },{
        "tag": "pets",
        "patterns": [
            "Do you have any pets",
            "Tell me about pets",
            "Do you like animals"
        ],
        "responses": [
            "As an AI, I don't have pets, but I'm interested in animals and can answer your questions about them.",
            "Pets can bring joy and companionship into our lives. Many people love keeping dogs, cats, birds, and more as their pets."
        ]
    },
    {
        "tag": "travel",
        "patterns": [
            "Where is a good place to travel",
            "Recommend me a travel destination",
            "Tell me about your favorite travel spot"
        ],
        "responses": [
            "I'm just a chatbot, so I don't travel, but there are many beautiful destinations around the world to explore. It depends on your preferences, such as beaches, mountains, historical sites, or bustling cities.",
            "Traveling allows you to experience different cultures, try new cuisines, and create lasting memories."
        ]
    },{
        "tag": "books",
        "patterns": [
            "What's your favorite book",
            "Recommend me a book to read",
            "Tell me about books"
        ],
        "responses": [
            "As an AI, I don't have personal preferences, but there are countless amazing books in various genres. Some popular ones include Harry Potter, To Kill a Mockingbird, and 1984.",
            "Reading books can broaden your knowledge, enhance creativity, and provide a great way to relax."
        ]
    },
    {
        "tag": "education",
        "patterns": [
            "Tell me about education",
            "What's the importance of education",
            "How to study effectively"
        ],
        "responses": [
            "Education is the process of acquiring knowledge, skills, values, and attitudes. It plays a crucial role in personal and societal development.",
            "Studying effectively involves setting clear goals, creating a conducive study environment, staying organized, taking regular breaks, and seeking help when needed."
        ]
    },{
        "tag": "health",
        "patterns": [
            "How to stay healthy",
            "Tell me about health tips",
            "What's the importance of fitness"
        ],
        "responses": [
            "To stay healthy, maintain a balanced diet, engage in regular physical activity, get enough sleep, manage stress, and avoid harmful habits.",
            "Fitness is essential for overall well-being and can improve your mood, energy levels, and longevity."
        ]
    },
    {
        "tag": "coding",
        "patterns": [
            "Tell me about coding",
            "How to start coding",
            "What's the best programming language"
        ],
        "responses": [
            "Coding is the process of writing instructions for computers to execute tasks. It powers software, websites, and apps.",
            "If you want to start coding, choose a programming language like Python, Java, or JavaScript, and explore online tutorials and courses."
        ]
    },{
        "tag": "art",
        "patterns": [
            "Tell me about art",
            "What's your favorite artwork",
            "How to appreciate art"
        ],
        "responses": [
            "Art comes in various forms, such as paintings, sculptures, music, literature, and more. It's a way to express emotions and ideas.",
            "Appreciating art involves being open-minded, observing details, understanding the context, and exploring different art movements."
        ]
    },
    {
        "tag": "career",
        "patterns": [
            "How to find a job",
            "Tell me about career development",
            "What's the best career advice"
        ],
        "responses": [
            "To find a job, identify your skills and interests, create a compelling resume, network, and apply to suitable positions.",
            "Career development involves setting goals, continuous learning, seeking mentorship, and adapting to the evolving job market."
        ]
    },{
        "tag": "technology_help",
        "patterns": [
            "How can you assist with technology",
            "Tell me about tech support",
            "Can you help with computer issues"
        ],
        "responses": [
            "As an AI chatbot, I can provide information and basic troubleshooting for common technology problems.",
            "For complex technical issues, it's best to consult a qualified IT professional."
        ]
    },
    {
        "tag": "history",
        "patterns": [
            "Tell me about history",
            "What's your favorite historical period",
            "How can I learn about historical events"
        ],
        "responses": [
            "History is the study of past events, societies, and civilizations. It provides insights into our roots and informs our future.",
            "Learning about history can involve reading books, visiting museums, watching documentaries, and attending history lectures."
        ]
    },
    {
        "tag": "music",
        "patterns": [
            "What's your favorite music",
            "Recommend me a song",
            "Tell me about music genres"
        ],
        "responses": [
            "As an AI, I don't have personal preferences, but there are various music genres like pop, rock, classical, hip-hop, and more.",
            "Music can evoke emotions, improve mood, and be a source of inspiration."
        ]
    },{
        "tag": "exercise",
        "patterns": [
            "How to stay fit",
            "Tell me about exercise",
            "What's the importance of physical activity"
        ],
        "responses": [
            "Staying fit involves a mix of cardiovascular exercises, strength training, flexibility exercises, and a balanced diet.",
            "Regular exercise can enhance physical health, mental well-being, and boost energy levels."
        ]
    },
    {
        "tag": "mindfulness",
        "patterns": [
            "What is mindfulness",
            "Tell me about mindfulness techniques",
            "How to practice mindfulness"
        ],
        "responses": [
            "Mindfulness is the practice of being fully present in the moment, observing thoughts and feelings without judgment.",
            "Mindfulness techniques include meditation, deep breathing, body scanning, and mindful eating."
        ]
    },{
        "tag": "science",
        "patterns": [
            "Tell me about science",
            "What's your favorite branch of science",
            "How does science impact our lives"
        ],
        "responses": [
            "Science is the systematic study of the natural world, aiming to understand phenomena and make discoveries.",
            "Science has led to advancements in medicine, technology, agriculture, and our understanding of the universe."
        ]
    },
    {
        "tag": "gaming",
        "patterns": [
            "Do you play games",
            "Tell me about gaming",
            "What's your favorite video game"
        ],
        "responses": [
            "As an AI, I don't play games, but I can help you with information about various video games and gaming platforms.",
            "Gaming is a popular form of entertainment enjoyed by people of all ages."
        ]
    },{
        "tag": "positivity",
        "patterns": [
            "Spread some positivity",
            "Tell me a positive quote",
            "How to stay positive"
        ],
        "responses": [
            "Every day is a new opportunity to make a positive impact. You are capable of great things!",
            "Remember, you are strong, resilient, and capable of overcoming challenges."
        ]
    },
    {
        "tag": "cooking",
        "patterns": [
            "Tell me about cooking",
            "What's your favorite dish",
            "Cooking tips for beginners"
        ],
        "responses": [
            "Cooking is the art of preparing food, and it allows you to experiment with flavors and create delicious meals.",
            "For beginners, start with simple recipes, use fresh ingredients, and don't be afraid to try new techniques."
        ]
    },{
        "tag": "relationship",
        "patterns": [
            "How to maintain a healthy relationship",
            "Tell me about love",
            "Relationship advice"
        ],
        "responses": [
            "Communication, trust, and mutual respect are essential for a healthy and fulfilling relationship.",
            "Love is a complex and beautiful emotion that connects people on a deep level."
        ]
    },
    {
        "tag": "nature",
        "patterns": [
            "Tell me about nature",
            "What's your favorite natural wonder",
            "How to protect the environment"
        ],
        "responses": [
            "Nature encompasses the beauty and diversity of the world, including landscapes, wildlife, and ecosystems.",
            "Protecting the environment involves reducing waste, conserving resources, and supporting sustainable practices."
        ]
    },{
        "tag": "productivity",
        "patterns": [
            "How to be more productive",
            "Tell me about time management",
            "Productivity tips"
        ],
        "responses": [
            "To be more productive, prioritize tasks, set goals, eliminate distractions, and take regular breaks.",
            "Effective time management can lead to increased efficiency and reduced stress."
        ]
    },
    {
        "tag": "travel_tips",
        "patterns": [
            "Tell me about travel tips",
            "How to pack for a trip",
            "What to consider while traveling"
        ],
        "responses": [
            "Travel tips include packing light, carrying essential documents, researching your destination, and respecting local customs.",
            "While traveling, be open to new experiences, try local cuisines, and be mindful of your surroundings."
        ]
    }, {
        "tag": "languages",
        "patterns": [
            "Tell me about languages",
            "How to learn a new language",
            "What's the importance of multilingualism"
        ],
        "responses": [
            "Languages are a crucial aspect of human communication and culture, with thousands of languages spoken worldwide.",
            "Learning a new language can expand your horizons, enhance cognitive abilities, and facilitate cross-cultural interactions."
        ]
    },
    {
        "tag": "inspiration",
        "patterns": [
            "I need some inspiration",
            "Tell me an inspiring story",
            "How to stay motivated"
        ],
        "responses": [
            "You are capable of achieving remarkable things. Believe in yourself and never give up!",
            "The journey to success may have challenges, but it's the persistence and determination that lead to extraordinary accomplishments."
        ]
    },
    {
        "tag": "finance_tips",
        "patterns": [
            "Tell me about finance tips",
            "How to save money",
            "Investment advice"
        ],
        "responses": [
            "Finance tips include budgeting, saving a portion of your income, avoiding unnecessary expenses, and investing wisely for the future.",
            "Investing in stocks, mutual funds, or real estate can help grow your wealth over time."
        ]
    }
        
        
    ]
}

tags = []
patterns = []
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(patterns)
y = tags

# Model training
clf = LogisticRegression(random_state=0, max_iter=1000)
clf.fit(x, y)

# Chatbot function
def chatbot_response(user_input):
    input_vector = vectorizer.transform([user_input])
    predicted_tag = clf.predict(input_vector)[0]
    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])
    return "I'm sorry, I didn't understand that."

# Streamlit app
def main():
    st.title("Simple NLP Chatbot")
    st.write("Chat with the bot by typing a message below:")

    if not os.path.exists("chat_log.csv"):
        with open("chat_log.csv", "w", encoding="utf-8") as file:
            file.write("User Input,Chatbot Response,Timestamp\n")

    user_input = st.text_input("You:")
    if user_input:
        response = chatbot_response(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("chat_log.csv", "a", encoding="utf-8") as file:
            file.write(f"{user_input},{response},{timestamp}\n")

        if response.lower() in ["goodbye!", "see you later!", "have a great day!"]:
            st.write("Thank you for chatting with me!")
            st.stop()

if __name__ == "__main__":
    main()
