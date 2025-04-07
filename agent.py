import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyttsx3
from perception_module import PerceptionModule
import os
import platform
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        self.perception = PerceptionModule()
        self.tts_engine = pyttsx3.init()
        self.current_voice_id = None  # Track the current voice ID
        # Determine the device to use
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Intrusive thoughts configuration
        self.intrusive_thoughts_enabled = False
        self.intrusive_thoughts_level = 0.3  # Default level (0.0 to 1.0)
        
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

    def speak(self, text):
        try:
            # Ensure the voice is set before speaking
            if self.current_voice_id:
                self.tts_engine.setProperty('voice', self.current_voice_id)
            self.tts_engine.stop()
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"[TTS Error] {e}")

    def list_available_voices(self):
        """List all available TTS voices on the system and let user select one.
        
        Returns:
            The selected voice ID
        """
        # Reinitialize engine to ensure we get a fresh list
        self.tts_engine = pyttsx3.init()
        voices = self.tts_engine.getProperty('voices')
        
        if not voices:
            print("No voices found on the system. Using default voice.")
            return None
            
        print("\nAvailable voices:")
        for idx, voice in enumerate(voices):
            # Print details about the voice
            gender = voice.gender if hasattr(voice, 'gender') else 'unknown'
            age = voice.age if hasattr(voice, 'age') else 'unknown'
            print(f"{idx+1}. {voice.name} ({gender}, {age})")
        
        selected_idx = 0
        while selected_idx < 1 or selected_idx > len(voices):
            try:
                selected_idx = int(input(f"\nSelect a voice (1-{len(voices)}): "))
            except ValueError:
                print("Please enter a valid number.")
        
        selected_voice = voices[selected_idx-1]
        print(f"Selected voice: {selected_voice.name}")
        return selected_voice.id
        
    def set_voice(self, voice_id):
        """Set the TTS voice to use.
        
        Args:
            voice_id: The ID of the voice to use
        """
        if not voice_id:
            print("No voice ID provided, using default voice")
            return
            
        print(f"Setting voice to {voice_id}")
        # Reinitialize the engine to ensure clean state
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('voice', voice_id)
        self.current_voice_id = voice_id

        # Test the voice to confirm it's working
        print("Testing selected voice...")
        self.tts_engine.say(f"Hello, just give me one second.")
        self.tts_engine.runAndWait()

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
                    do_sample=True,
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
                    max_new_tokens=80,
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
    
    def generate_intrusive_thoughts(self, conversation_history):
        """Generate slightly modified conversation history to simulate intrusive thoughts.
        
        Args:
            conversation_history: The original conversation history
            
        Returns:
            Modified conversation history with slight changes
        """
        if not self.intrusive_thoughts_enabled or self.intrusive_thoughts_level <= 0.0:
            return conversation_history
        
        # Split the conversation into individual messages
        messages = []
        current_message = []
        current_speaker = None
        
        # Parse the conversation history into separate messages
        for line in conversation_history.split('\n'):
            if line.startswith('<|user|>'):
                if current_speaker:
                    messages.append((current_speaker, '\n'.join(current_message)))
                current_speaker = 'user'
                current_message = []
            elif line.startswith('<|assistant|>'):
                if current_speaker:
                    messages.append((current_speaker, '\n'.join(current_message)))
                current_speaker = 'assistant'
                current_message = []
            else:
                current_message.append(line)
        
        # Add the last message if it exists
        if current_speaker and current_message:
            messages.append((current_speaker, '\n'.join(current_message)))
        
        # Nothing to modify
        if not messages:
            return conversation_history
        
        # Identify user messages only
        user_indices = [i for i, (speaker, _) in enumerate(messages) if speaker == 'user']
        if not user_indices:
            return conversation_history
        
        # Calculate how many messages to modify based on intrusive thoughts level
        # Level 0.1 = 10% of user messages, level 1.0 = 100% of user messages
        num_to_modify = max(1, int(len(user_indices) * self.intrusive_thoughts_level))
        
        # Randomly select which user messages to modify
        import random
        indices_to_modify = random.sample(user_indices, min(num_to_modify, len(user_indices)))
        
        # Debug information
        self.debug_print(f"INTRUSIVE THOUGHTS (level {self.intrusive_thoughts_level}):")
        self.debug_print(f"Modifying {num_to_modify} out of {len(user_indices)} user messages.")
        
        # For each selected message, generate a modified version
        for idx in indices_to_modify:
            speaker, content = messages[idx]
            
            # Level determines how much to modify
            level_descriptor = "very slightly" if self.intrusive_thoughts_level < 0.3 else \
                              "somewhat" if self.intrusive_thoughts_level < 0.6 else \
                              "significantly"
            
            prompt = f"""<|system|>
You are simulating intrusive thoughts for a mental health coach. You need to {level_descriptor} modify the following user message.
Make only subtle changes that could represent random thoughts or slight misinterpretations.

Intensity level: {self.intrusive_thoughts_level * 10}/10

Guidelines:
1. Change details that could be misremembered or misinterpreted
2. You may slightly alter emotions, intentions, or specific words
3. Do not change the main topic or completely rewrite the message
4. The higher the intensity level, the more noticeable the changes should be
5. Preserve the original message length and structure

Original message:
{content}

Return ONLY the modified message with your subtle changes. Do not include any explanation, commentary, or additional text.
<|assistant|>
"""
            
            # Generate modified message
            inputs = self.tokenizer(prompt, return_tensors='pt', padding=True)
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            
            try:
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=len(content.split()) * 2,  # Allow twice the token count for modification
                        do_sample=True,
                        temperature=0.7,  # Some randomness in the modifications
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # Decode only the newly generated tokens
                generated_ids = output_ids[0, input_ids.shape[1]:]
                modified_message = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                # Check if we have a valid modified message
                if len(modified_message) < 5:
                    # Failed to generate proper intrusive thought, keep original
                    self.debug_print(f"Failed to modify message {idx}. Using original.")
                else:
                    self.debug_print(f"Message {idx} modified:")
                    self.debug_print(f"Original: {content}")
                    self.debug_print(f"Modified: {modified_message}")
                    # Update the message with the modified content
                    messages[idx] = (speaker, modified_message)
                    
            except Exception as e:
                print(f"Error modifying message {idx}: {e}")
        
        # Reconstruct the conversation with modified messages
        modified_history = ""
        for speaker, content in messages:
            if speaker == 'user':
                modified_history += f"<|user|>\n{content}\n"
            else:
                modified_history += f"<|assistant|>\n{content}\n"
        
        return modified_history
    
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
        original_chat_history = self.memory.get_formatted_history()
        
        # If intrusive thoughts are enabled, generate a modified conversation history
        if self.intrusive_thoughts_enabled:
            chat_history = self.generate_intrusive_thoughts(original_chat_history)
        else:
            chat_history = original_chat_history
        
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
        
        # Set appropriate token limits based on mode - increased for more substantial responses
        adjusted_max_tokens = 250 if mode == "guidance" else 150
        
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
        # ANSI color codes for colored text
        LIGHT_BLUE = "\033[94m"
        RESET = "\033[0m"  # Reset all formatting
        
        print(f"{self.coach_name} Mental Health Coach Chat")
        print("Type 'quit' to exit the conversation.\n")
        
        # Set up intrusive thoughts
        enable_intrusive = input("Enable intrusive thoughts? (y/n): ").lower().strip() == 'y'
        self.intrusive_thoughts_enabled = enable_intrusive
        
        if self.intrusive_thoughts_enabled:
            try:
                level = float(input("Enter intrusive thoughts level (0.1-1.0): "))
                self.intrusive_thoughts_level = min(1.0, max(0.1, level))
                print(f"Intrusive thoughts enabled at level {self.intrusive_thoughts_level}")
            except ValueError:
                self.intrusive_thoughts_level = 0.3
                print(f"Using default intrusive thoughts level: {self.intrusive_thoughts_level}")
        
        # Allow user to select a TTS voice
        voice_id = self.list_available_voices()
        self.set_voice(voice_id)
        
        # Start the conversation with the coach speaking first
        initial_greeting = self.get_initial_greeting()
        formatted_greeting = self.format_for_terminal(initial_greeting)
        print(f"{self.coach_name}: {LIGHT_BLUE}{formatted_greeting}{RESET}")
        self.speak(initial_greeting)

        while True:
            transcript, emotion = self.perception.percieve_combined()
            if transcript.lower() in ["quit", "exit", 'goodbye', 'good bye']:
                self.speak("Take care of yourself. Goodbye!")
                break
                
            # Add user input to memory with emotion noted
            self.memory.add_message("User", f"[Emotion: {emotion}] {transcript}")
            response = self.generate_response(transcript)
            formatted_response = self.format_for_terminal(response)
            print(f"{self.coach_name}: {LIGHT_BLUE}{formatted_response}{RESET}")
            self.speak(response)

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