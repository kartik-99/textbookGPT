# PDF-based Conversational Chatbot using LangChain

This project showcases a PDF-based conversational chatbot built using Python and LangChain. The chatbot can read any textbook in PDF format and help users understand the concepts explained in it. The application uses OpenAI's GPT API to process questions and maintain chat history!

With more leasure time, I will try to add more features/optimisations

## Features

- Extracts text from a PDF file.
- Splits the extracted text into manageable chunks.
- Stores text chunks in a Chroma vector database.
- Uses LangChain to create a conversational retrieval chain.
- Maintains chat history to provide contextually relevant responses.

## Prerequisites

- Python 3.7 or higher
- `pip` for package installation
- An OpenAI API key

## Installation

1. **Clone the Repository**

   ```sh
   git clone https://github.com/yourusername/pdf-conversational-chatbot.git
   cd pdf-conversational-chatbot
   ```

2. **Create a Virtual Environment**

   ```sh
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install the Dependencies**

   ```sh
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**

   Create a `.env` file in the root directory of the project and add your OpenAI API key:

   ```sh
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

1. **Run the Application**

   ```sh
   python main.py
   ```

2. **Enter the Path to Your PDF File**

   When prompted, enter the path to the PDF file you want to use.

   ```sh
   Enter the path to the PDF file: /path/to/your/textbook.pdf
   ```

3. **Ask Questions**

   Once the PDF is loaded, you can start asking questions about the content of the PDF. Type your questions in the terminal and press Enter.

   ```sh
   You: What topics are covered in this book?
   TextBookGPT: The book covers topics such as...
   ```

4. **Exit the Application**

   To exit the application, type `exit` or `quit` and press Enter.

   ```sh
   You: exit
   ```

## Acknowledgements

- [OpenAI](https://openai.com/) for their powerful GPT API.
- [LangChain](https://langchain.com/) for providing the tools to create conversational chains and good docs!
- [PyPDF2](https://pythonhosted.org/PyPDF2/) for PDF text extraction.