import speech_recognition as sr
import logging
from typing import Tuple, Optional, List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Handles voice input capture and transcription.
    """
    
    def __init__(self):
        """Initialize the speech recognizer."""
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 300
        self._available_devices = None
        
    def get_available_devices(self) -> List[Dict[str, any]]:
        """
        Get a list of available audio input devices.
        
        Returns:
            List of dictionaries containing device information
        """
        if self._available_devices is None:
            try:
                # Get list of microphone names
                mic_names = sr.Microphone.list_microphone_names()
                
                # Create list of device dictionaries
                self._available_devices = [
                    {"index": i, "name": name} 
                    for i, name in enumerate(mic_names)
                ]
                
                # If no devices found, return an empty list
                if not self._available_devices:
                    logger.warning("No microphone devices found")
                    self._available_devices = []
            except Exception as e:
                logger.error(f"Error getting microphone list: {e}")
                self._available_devices = []
                
        return self._available_devices
    
    def capture_and_transcribe(self, timeout: int = 5, device_index: Optional[int] = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Capture audio input and transcribe it to text.
        
        Args:
            timeout: Number of seconds to listen for
            device_index: Index of the microphone device to use (None for default)
            
        Returns:
            Tuple containing:
            - Success flag (boolean)
            - Transcribed text (if successful, otherwise None)
            - Error message (if unsuccessful, otherwise None)
        """
        try:
            # Use the specified device if provided, otherwise use default
            if device_index is not None:
                logger.info(f"Using microphone device with index {device_index}")
                microphone = sr.Microphone(device_index=device_index)
            else:
                logger.info("Using default microphone device")
                microphone = sr.Microphone()
                
            with microphone as source:
                logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                logger.info(f"Listening for {timeout} seconds...")
                audio = self.recognizer.listen(source, timeout=timeout)
                
                logger.info("Transcribing audio...")
                text = self.recognizer.recognize_google(audio)
                logger.info(f"Transcribed: {text}")
                
                return True, text, None
                
        except sr.WaitTimeoutError:
            return False, None, "🔴 NO SPEECH DETECTED 🔴 PLEASE TRY SPEAKING AGAIN OR USE TEXT INPUT INSTEAD"
        except sr.UnknownValueError:
            return False, None, "YOUR SPEECH WASN'T CLEAR. PLEASE TRY AGAIN WITH A CLEARER VOICE."
        except sr.RequestError as e:
            return False, None, f"SPEECH RECOGNITION SERVICE UNAVAILABLE. PLEASE USE TEXT INPUT INSTEAD."
        except Exception as e:
            logger.error(f"Unexpected error in audio processing: {e}")
            return False, None, f"AN ERROR OCCURRED WITH SPEECH RECOGNITION: {str(e)}"