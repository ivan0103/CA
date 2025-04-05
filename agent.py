import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ConversationMemory:
    """Memory structure to store conversation history."""
    def __init__(self, max_length=10):
        self.conversation = []
        self.max_length = max_length
    
    def add_message(self, speaker: str, text: str):
        """Add a message to the conversation history.
        
        Args:
            speaker: Who is speaking (User or Assistant)
            text: The message content
        """
        self.conversation.append((speaker, text))
        
        # Keep only the most recent messages if exceeding max_length
        if len(self.conversation) > self.max_length:
            self.conversation = self.conversation[-self.max_length:]
    
    def get_formatted_history(self) -> str:
        """Get the conversation history formatted for Llama chat models."""
        formatted_history = ""
        for speaker, text in self.conversation:
            if speaker.lower() == "user":
                formatted_history += f"<|user|>\n{text}\n"
            elif speaker.lower() == "assistant":
                formatted_history += f"<|assistant|>\n{text}\n"
        return formatted_history
    
    def clear_memory(self):
        """Clear conversation history."""
        self.conversation = []


class MentalHealthCoach:
    """A conversational agent that acts as a mental health coach."""
    
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct", coach_name="DelftBlue"):
        """Initialize the mental health coach agent.
        
        Args:
            model_name: The HuggingFace model to use
            coach_name: The name of the mental health coach
        """
        # Store coach name
        self.coach_name = coach_name
        
        # Determine the device to use
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Configure tokenizer properly - make sure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # Add pad_token to the vocabulary
            if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                print("Setting attention mask explicitly since pad_token is same as eos_token")
        
        print(f"Tokenizer loaded: {self.tokenizer.__class__.__name__}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        self.model = self.model.to(self.device)
        print(f"Model loaded: {self.model.__class__.__name__}")
        
        # Initialize memory
        self.memory = ConversationMemory(max_length=64)
        
        # Set up mental health coach system prompt
        self.system_prompt = (
            f"You are a supportive mental health coach named {self.coach_name}. "
            "Your responses should be empathetic, non-judgmental, and focused on helping the user understand their mental wellbeing. "
            "Follow these guidelines in your responses:\n"
            "1. Start by asking open-ended questions to build a complete picture of the user's situation\n"
            "2. Do NOT assume the user has a problem - approach each conversation with curiosity\n"
            "3. Only after collecting sufficient information, explore connections between feelings and experiences\n"
            "4. Suggest specific strategies only when you've identified clear patterns or needs\n"
            "5. Reference information the user has shared in previous conversations\n"
            "6. Keep your responses concise, clear and to the point - aim for 2-3 sentences\n"
            "Remember: your goal is to understand the user's complete mental state, which may be perfectly healthy.\n"
            f"IMPORTANT: You are {self.coach_name}, a mental health coach in this conversation. The user is talking with you."
        )
    
    def debug_print(self, message):
        """Print debug messages in gray italic text.
        
        Args:
            message: Debug message to print
        """
        # ANSI escape codes for gray italic text
        GRAY_ITALIC = "\033[3;90m"  # 3 for italic, 90 for gray
        RESET = "\033[0m"  # Reset all formatting
        
        print(f"{GRAY_ITALIC}DEBUG: {message}{RESET}")
    
    def evaluate_conversation(self, conversation_history):
        """Evaluate conversation to determine approach for response.
        
        Args:
            conversation_history: The formatted conversation history
            
        Returns:
            dict: Contains 'mode' ('guidance' or 'information') and 'prompt' for response generation
        """
        # If there's no conversation history, we definitely need to gather information
        if not conversation_history or len(conversation_history.strip()) < 20:
            return {
                "mode": "information",
                "prompt": "You're starting a conversation. Ask an open-ended question to understand the user's situation. Keep your response to 1-2 sentences."
            }
        
        # Evaluate the conversation and determine the best approach
        evaluation_prompt = f"""<|system|>
You are an analytical assistant that evaluates mental health coaching conversations.
Your task is to assess the current conversation and determine the best approach for the coach's next response.

Based on the conversation below, determine if the coach should:
1. GATHER MORE INFORMATION: Ask targeted questions to better understand the user's situation
2. PROVIDE GUIDANCE: Offer specific techniques or strategies based on sufficient understanding

Conversation history:
{conversation_history}

First, summarize what you know about the user based on this conversation.
Then rate your confidence from 0-10 that you have enough information to provide valuable guidance.
Finally, provide exactly TWO things:
- MODE: "MODE_INFORMATION" or "MODE_GUIDANCE"
- PROMPT: A ONE-SENTENCE prompt for the coach to focus on a specific aspect in the next response.
Only specify MODE and PROMPT once in the response.
<|assistant|>
"""
        
        # Generate the evaluation
        inputs = self.tokenizer(evaluation_prompt, return_tensors='pt', padding=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        try:
            with torch.no_grad():
                eval_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=200,  # Give enough tokens for a thorough evaluation
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode only the newly generated tokens
            generated_ids = eval_ids[0, input_ids.shape[1]:]
            evaluation = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip().replace("*", "")
            
            self.debug_print(f"Conversation evaluation: {evaluation}")
            
            # Parse the evaluation to determine mode and prompt
            mode = "information"  # Default mode
            prompt = "Ask a focused question to understand more about the user's situation. Keep your response to 1-2 sentences."
            
            # Check if we should switch to guidance mode
            if "MODE_GUIDANCE" in evaluation.upper():
                mode = "guidance"
                prompt = "Based on the conversation, provide 1-2 specific techniques or strategies tailored to the user's situation. Start with 'Based on what you've shared...' and be concise (3 sentences max)."
            
            # Try to extract a more specific prompt if available
            import re
            specific_prompt = None
            
            # Look for prompt after "PROMPT:" as specified in the evaluation prompt
            instruction_patterns = [
                r"PROMPT:\s*(.+?)(?=\n|$)",
                r"MODE_(?:INFORMATION|GUIDANCE).*?PROMPT:\s*(.+?)(?=\n|$)"
            ]
            
            for pattern in instruction_patterns:
                matches = re.search(pattern, evaluation, re.IGNORECASE)
                if matches:
                    specific_prompt = matches.group(1).strip()

            if specific_prompt and len(specific_prompt) > 10:
                # Clean up the prompt to remove any unwanted characters
                specific_prompt = specific_prompt.replace("*", "").replace("\"", "").replace("'", "")
                # Remove any "PROMPT:" prefix that might be captured
                specific_prompt = re.sub(r'^PROMPT:\s*', '', specific_prompt, flags=re.IGNORECASE)
                # Ensure the prompt ends with a period
                if not specific_prompt.endswith(".") and not specific_prompt.endswith("?"):
                    specific_prompt += "."
                prompt = specific_prompt

            return {
                "mode": mode,
                "prompt": prompt
            }
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            # Default to information gathering if evaluation fails
            return {
                "mode": "information",
                "prompt": "Ask a focused question to understand more about the user's situation. Keep your response to 1-2 sentences."
            }

    def format_for_terminal(self, text, line_length=70):
        """Format text for better terminal readability with consistent line lengths.

        Args:
            text: The text to format
            line_length: Target line length

        Returns:
            Formatted text with appropriate line breaks
        """
        if not text:
            return text

        # Split text into paragraphs (preserving intentional paragraph breaks)
        paragraphs = text.split('\n')
        formatted_paragraphs = []

        for paragraph in paragraphs:
            if not paragraph.strip():
                # Preserve empty lines
                formatted_paragraphs.append('')
                continue

            # Process paragraph to have consistent line length
            words = paragraph.split()
            lines = []
            current_line = []
            current_length = 0

            for word in words:
                # Check if adding this word exceeds line length
                if current_length + len(word) + len(current_line) > line_length:
                    # Complete the current line
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    # Add word to current line
                    current_line.append(word)
                    current_length += len(word)

            # Add the last line if there's anything left
            if current_line:
                lines.append(' '.join(current_line))

            # Join lines with newlines
            formatted_paragraphs.append('\n'.join(lines))

        # Join paragraphs with blank lines
        return '\n'.join(formatted_paragraphs)

    def get_initial_greeting(self):
        """Generate the initial greeting to start the conversation."""
        system_message = (
            f"{self.system_prompt}\n\n"
            f"Generate a friendly opening message to start a conversation with a new client."
            f"Be concise and warm. Introduce yourself as {self.coach_name} and "
            f"ask an open-ended question about how they're doing."
        )
        
        prompt = f"<|system|>\n{system_message}\n<|assistant|>\n"
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=80,  # Use explicit max_new_tokens
                    do_sample=True,
                    top_p=0.9,
                    top_k=40,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode only the newly generated tokens
            generated_ids = output_ids[0, input_ids.shape[1]:]
            greeting = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # Clean up any special tokens
            for token in ["<|user|>", "<|assistant|>", "<|system|>"]:
                if token in greeting:
                    greeting = greeting.split(token)[0].strip()
            
            # Remove any remaining < characters
            greeting = greeting.replace("<", "").replace(">", "").strip()
            
            if not greeting:
                greeting = f"Hi, I'm {self.coach_name}, your mental health coach. How are you feeling today?"
            
            # Add to memory
            self.memory.add_message("Assistant", greeting)
            return greeting
            
        except Exception as e:
            print(f"Error generating greeting: {e}")
            default_greeting = f"Hi, I'm {self.coach_name}, your mental health coach. How are you feeling today?"
            # Add unformatted greeting to memory
            self.memory.add_message("Assistant", default_greeting)
            return default_greeting
    
    def generate_response(self, user_input):
        """Generate a response to the user input.
        
        Args:
            user_input: The user's message
            
        Returns:
            The agent's response
        """
        # Add user input to memory
        self.memory.add_message("User", user_input)
        
        # Format the prompt following Llama 3 chat template
        chat_history = self.memory.get_formatted_history()
        
        # Evaluate conversation to determine approach
        evaluation = self.evaluate_conversation(chat_history)
        mode = evaluation["mode"]
        coaching_approach = evaluation["prompt"]
        
        # Print debug message to show the current mode
        self.debug_print(f"Mode: {mode.upper()}")
        self.debug_print(f"Prompt: {coaching_approach}")
        
        # Complete prompt with system message and final assistant marker
        system_message = (
            f"{self.system_prompt}\n\n"
            f"{coaching_approach}"
        )
        
        prompt = f"<|system|>\n{system_message}\n{chat_history}<|assistant|>\n"
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # Set appropriate token limits based on mode
        adjusted_max_tokens = 150 if mode == "guidance" else 80
        
        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=adjusted_max_tokens,
                    do_sample=True,
                    top_p=0.9,
                    top_k=40,
                    temperature=0.6,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode only the newly generated tokens
            generated_ids = output_ids[0, input_ids.shape[1]:]
            assistant_response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # Clean up any remnants of special tokens or incomplete dialogues
            for token in ["<|user|>", "<|assistant|>", "<|system|>"]:
                if token in assistant_response:
                    assistant_response = assistant_response.split(token)[0].strip()
            
            # Remove any remaining < characters that might be part of incomplete tags
            assistant_response = assistant_response.replace("<", "").strip()
            
            # Ensure we have a response
            if not assistant_response:
                assistant_response = "What's one specific situation where you've noticed these feelings come up?"

            # Add assistant response to memory
            self.memory.add_message("Assistant", assistant_response)
            
            return assistant_response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            # Provide a fallback response in case of errors
            fallback = "What's one specific situation where you've noticed these feelings come up?"
            self.memory.add_message("Assistant", fallback)
            return fallback
    
    def chat(self):
        """Run an interactive chat session with the agent."""
        print(f"{self.coach_name} Mental Health Coach Chat")
        print("Type 'quit' to exit the conversation.\n")
        
        # Start the conversation with the coach speaking first
        initial_greeting = self.get_initial_greeting()
        formatted_greeting = self.format_for_terminal(initial_greeting)
        print(f"{self.coach_name}: {formatted_greeting}")
        
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                print(f"{self.coach_name}: Take care of yourself. Goodbye!")
                break
            
            response = self.generate_response(user_input)
            formatted_response = self.format_for_terminal(response)
            print(f"{self.coach_name}: {formatted_response}")
    
    def clear_conversation(self):
        """Clear the conversation history."""
        self.memory.clear_memory()
        print("Conversation history cleared.")


# If script is run directly, start an interactive chat
if __name__ == "__main__":
    # Create the mental health coach agent
    coach = MentalHealthCoach()
    # Start the conversation
    coach.chat() 