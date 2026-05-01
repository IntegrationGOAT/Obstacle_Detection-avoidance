import pyttsx3
import time
from threading import Thread


class AudioHelper:
    """Handles text-to-speech guidance with throttling and caching."""
    
    def __init__(self, rate=160):
        """Initialize the text-to-speech engine.
        
        Args:
            rate: Speech rate (words per minute)
        """
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.available = True
        except Exception as e:
            print(f"Warning: TTS engine not available: {e}")
            self.available = False
        
        self.last_spoken = ""
        self.last_time = 0
        self.memory = {}  # Tracks last spoken time per message
    
    def should_speak(self, text, min_interval=2, message_cooldown=5):
        """Determine if a message should be spoken based on throttling rules.
        
        Args:
            text: The text to potentially speak
            min_interval: Minimum seconds between any speech (0-1 range: 2 sec)
            message_cooldown: Minimum seconds before repeating same message (0-1 range: 5 sec)
            
        Returns:
            bool: True if message should be spoken
        """
        current_time = time.time()
        
        # Check if enough time passed since last speech
        if (current_time - self.last_time) < min_interval:
            return False
        
        # Check if message changed
        if text == self.last_spoken:
            return False
        
        # Check if same message spoke recently
        if text in self.memory and (current_time - self.memory[text]) < message_cooldown:
            return False
        
        return True
    
    def speak(self, text, async_mode=False):
        """Speak the given text.
        
        Args:
            text: The text to speak
            async_mode: If True, speak in a background thread (doesn't block)
            
        Returns:
            bool: True if speech was triggered, False if throttled
        """
        if not self.available:
            return False
        
        if not self.should_speak(text):
            return False
        
        current_time = time.time()
        
        def speak_thread():
            try:
                self.engine.stop()
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Error speaking: {e}")
        
        if async_mode:
            thread = Thread(target=speak_thread, daemon=True)
            thread.start()
        else:
            speak_thread()
        
        # Update memory
        self.memory[text] = current_time
        self.last_spoken = text
        self.last_time = current_time
        
        return True
    
    def stop(self):
        """Stop any ongoing speech."""
        if self.available:
            try:
                self.engine.stop()
            except Exception as e:
                print(f"Error stopping speech: {e}")
    
    def reset_memory(self):
        """Clear the message memory (allows immediate repetition)."""
        self.memory.clear()
        self.last_spoken = ""
        self.last_time = 0
