import queue
import threading
import sounddevice as sd
import numpy as np
import torch
import chromadb
import google.generativeai as genai
from tavily import TavilyClient
import gradio as gr
import soundfile as sf
from faster_whisper import WhisperModel
from typing import List, Dict, Any
import concurrent.futures
import os
import pyaudio
import wave


class GoogleMeetAudioCapture:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stop_event = threading.Event()
        self.recording_thread = None
        
        # PyAudio configuration for system audio capture
        self.pyaudio_interface = pyaudio.PyAudio()
        self.stream = None

    def list_audio_devices(self):
        """List available audio devices for system audio capture"""
        device_count = self.pyaudio_interface.get_device_count()
        devices = []
        for i in range(device_count):
            device_info = self.pyaudio_interface.get_device_info_by_index(i)
            devices.append({
                'index': i,
                'name': device_info['name'],
                'max_input_channels': device_info['maxInputChannels'],
                'max_output_channels': device_info['maxOutputChannels']
            })
        return devices

    def start_recording(self, device_index=None):
        """
        Start recording system audio from Google Meet
        
        :param device_index: Specific audio device index to capture from 
                             (use list_audio_devices() to find the right index)
        """
        if self.is_recording:
            print("Recording is already in progress.")
            return

        self.is_recording = True
        self.stop_event.clear()

        def recording_worker():
            # Open the stream for system audio capture
            self.stream = self.pyaudio_interface.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,  # Use specified device or default
                frames_per_buffer=1024
            )

            # Recording loop
            while not self.stop_event.is_set():
                try:
                    # Read audio data
                    audio_data = self.stream.read(1024)
                    # Convert to numpy array
                    np_audio = np.frombuffer(audio_data, dtype=np.int16)
                    # Put in queue
                    self.audio_queue.put(np_audio)
                except Exception as e:
                    print(f"Audio capture error: {e}")
                    break

        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=recording_worker)
        self.recording_thread.start()
        print("Started system audio recording.")

    def stop_recording(self):
        """Stop the audio recording"""
        if not self.is_recording:
            print("No recording in progress.")
            return

        # Signal to stop recording
        self.stop_event.set()

        # Close PyAudio stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        # Wait for recording thread to finish
        if self.recording_thread:
            self.recording_thread.join()

        # Reset recording state
        self.is_recording = False
        print("Stopped system audio recording.")

    def save_recorded_audio(self, filename='google_meet_recording.wav'):
        """
        Save recorded audio chunks to a WAV file
        
        :param filename: Output filename for the audio recording
        """
        # Collect all audio chunks
        audio_chunks = []
        while not self.audio_queue.empty():
            audio_chunks.append(self.audio_queue.get())

        # Concatenate chunks
        if audio_chunks:
            audio_data = np.concatenate(audio_chunks)
            
            # Save to WAV file
            sf.write(filename, audio_data, self.sample_rate)
            print(f"Audio saved to {filename}")
            return filename
        else:
            print("No audio recorded.")
            return None


class AgenticRAGSystem:
    def __init__(self):
        # LLM Configuration
        genai.configure(api_key="AIzaSyB1luPxiGaDFNYazTFZMWlwz1Qrg_11Vn8")
        self.llm = genai.GenerativeModel('gemini-1.5-pro')
        
        # Web Search Configuration
        self.tavily = TavilyClient(api_key="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")
        
        # Vector Database Configuration
        self.chroma_client = chromadb.Client()
        try:
            self.collection = self.chroma_client.get_collection("knowledge_base")
        except:
            self.collection = self.chroma_client.create_collection("knowledge_base")

    def extract_implicit_question(self, transcription: str) -> str:
        """
        Advanced implicit question extraction using Chain of Thought
        """
        cot_prompt = f"""Carefully analyze the following transcription to extract the most meaningful implicit question:

Transcription: '{transcription}'

Comprehensive Analysis Process:
1. Identify Core Topics
   - What are the main subjects discussed?
   - Are there any unclear or ambiguous statements?

2. Knowledge Gap Identification
   - What specific information seems to be missing?
   - What would someone want to know more about after hearing this?

3. Question Formulation
   - Transform the implied information need into a precise, answerable question
   - Ensure the question is specific and focused

4. Prioritization
   - Select the most significant question that represents the key insight or curiosity in the transcription

Output Guidelines:
- Provide a single, clear, concise question
- The question should be directly derivable from the transcription
- Avoid broad or overly general questions
- Focus on extracting the most nuanced and meaningful query"""

        try:
            response = self.llm.generate_content(cot_prompt)
            implicit_question = response.text.strip()
            
            # Additional validation to ensure a meaningful question
            if len(implicit_question) < 10 or len(implicit_question) > 200:
                return "What key information is missing from this context?"
            
            return implicit_question
        except Exception as e:
            print(f"Question extraction error: {e}")
            return "What key information is missing from this context?"

    def decision_making_agent(self, question: str, transcription: str) -> str:
        """
        Advanced tool-calling agent with multi-dimensional decision making
        """
        decision_prompt = f"""Determine the OPTIMAL retrieval strategy for this question:

Question: '{question}'
Original Context: '{transcription}'

Comprehensive Decision Matrix:

1. Information Complexity
   - How specialized is the knowledge required?
   - Does it need domain-specific expertise?

2. Temporal Relevance
   - Is this about current events?
   - Does it require up-to-date information?

3. Source Reliability
   - Can internal knowledge (RAG) sufficiently answer?
   - Might external web sources provide better insight?

4. Query Characteristics
   - Factual or analytical?
   - Requires deep explanation or quick facts?

Retrieval Strategy Options:
a) RAG: For deep, specialized, context-rich queries
b) WebSearch: For current events, broad research, recent information
c) DirectLLM: For general knowledge, interpretative questions

Decision Criteria Weights:
- Domain Specificity: High
- Temporal Sensitivity: Moderate
- Complexity: High

Provide ONLY the strategy name: RAG, WebSearch, or DirectLLM

Reasoning should demonstrate a nuanced understanding of the query's nature."""

        try:
            strategy_response = self.llm.generate_content(decision_prompt)
            strategy = strategy_response.text.strip()
            
            valid_strategies = ['RAG', 'WebSearch', 'DirectLLM']
            return strategy if strategy in valid_strategies else 'DirectLLM'
        
        except Exception as e:
            print(f"Strategy decision error: {e}")
            return 'DirectLLM'

    def rag_retrieval(self, question: str) -> str:
        """
        Enhanced RAG Retrieval with more intelligent context matching
        """
        try:
            # More sophisticated retrieval using semantic similarity
            retrieved_docs = self.collection.query(
                query_texts=[question],
                n_results=3
            )
            
            context = "\n".join(retrieved_docs.get('documents', []))
            
            rag_prompt = f"""Synthesize a comprehensive answer using the following context:

Precise Question: {question}
Retrieved Context: {context}

Analysis Guidelines:
1. Directly address the specific question
2. Use retrieved context to provide a well-supported answer
3. If context is insufficient, clearly explain what additional information is needed
4. Prioritize accuracy and relevance over verbosity"""
            
            response = self.llm.generate_content(rag_prompt)
            return response.text
        
        except Exception as e:
            print(f"RAG retrieval error: {e}")
            return "Specialized retrieval failed. Switching to alternative strategy."

    def web_search(self, question: str) -> str:
        """
        Precise web search with advanced result synthesis
        """
        try:
            # Use the extracted implicit question for more precise search
            search_results = self.tavily.search(query=question, max_results=3)
            
            synthesis_prompt = f"""Synthesize a comprehensive, authoritative answer from these web search results:

Precise Search Query: '{question}'
Search Results: {search_results}

Synthesis Instructions:
1. Extract core factual information
2. Cross-reference multiple sources
3. Provide a clear, concise, and informative response
4. Include key insights from the search results
5. Maintain academic rigor in presentation"""
            
            summary = self.llm.generate_content(synthesis_prompt)
            return summary.text
        
        except Exception as e:
            print(f"Web search error: {e}")
            return "Web search encountered challenges. Falling back to direct knowledge retrieval."

    def direct_llm_response(self, question: str) -> str:
        """
        Intelligent direct LLM response generation
        """
        try:
            comprehensive_prompt = f"""Provide a thoughtful, comprehensive response to the following query:

Question: {question}

Response Guidelines:
1. Draw from broad knowledge base
2. Provide nuanced, multi-perspective insights
3. Maintain clarity and intellectual depth
4. If uncertain, acknowledge limitations
5. Prioritize informative and balanced explanation"""
            
            response = self.llm.generate_content(comprehensive_prompt)
            return response.text
        
        except Exception as e:
            print(f"Direct LLM response error: {e}")
            return "Unable to generate a comprehensive response at this time."

    def process_transcription(self, transcription: str) -> Dict[str, str]:
        """
        Orchestration method with enhanced decision-making
        """
        # Step 1: Extract Implicit Question
        implicit_question = self.extract_implicit_question(transcription)
        
        # Step 2: Decide Retrieval Strategy
        strategy = self.decision_making_agent(implicit_question, transcription)
        
        # Step 3: Execute Chosen Strategy
        if strategy == 'RAG':
            answer = self.rag_retrieval(implicit_question)
        elif strategy == 'WebSearch':
            answer = self.web_search(implicit_question)
        else:
            answer = self.direct_llm_response(implicit_question)
        
        return {
            'transcription': transcription,
            'question': implicit_question,
            'strategy': strategy,
            'answer': answer
        }


class AudioCapture:
    def __init__(self, sample_rate=16000, channels=1, device=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None
        self.recording_thread = None
        self.stop_event = threading.Event()

    def start_recording(self):
        self.is_recording = True
        self.stop_event.clear()
        channels = 1
        
        def recording_worker():
            with sd.InputStream(
                samplerate=self.sample_rate, 
                channels=self.channels,
                callback=self._audio_callback,
                device=self.device
            ) as stream:
                while not self.stop_event.is_set():
                    sd.sleep(100)

        self.recording_thread = threading.Thread(target=recording_worker)
        self.recording_thread.start()

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        if self.is_recording:
            self.audio_queue.put(indata.copy())

    def stop_recording(self):
        self.is_recording = False
        self.stop_event.set()
        if self.recording_thread:
            self.recording_thread.join()

    def get_audio_chunks(self):
        chunks = []
        while not self.audio_queue.empty():
            chunks.append(self.audio_queue.get())
        return np.concatenate(chunks) if chunks else np.array([])

class TranscriptionHandler:
    # [Modify TranscriptionHandler to work with system audio capture]
    def __init__(self):
        # Initialize Faster Whisper model with medium English model
        self.stt_model = WhisperModel(
            "medium.en", 
            device="cuda",  # Use "cpu" if no GPU available
            compute_type="float16"
        )
        
        self.transcription_queue = queue.Queue()
        self.transcription_history = []
        self.transcription_lock = threading.Lock()

    def transcribe_audio(self, audio_file):
        try:
            # Transcribe using Faster Whisper
            segments, info = self.stt_model.transcribe(
                audio_file, 
                beam_size=5,
                language='en'
            )
            
            # Combine transcribed segments
            transcription = " ".join([segment.text for segment in segments])
            
            if transcription.strip():
                with self.transcription_lock:
                    self.transcription_queue.put(transcription)
                    self.transcription_history.append(transcription)
            
            return transcription
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

class GoogleMeetCompanion:
    def __init__(self):
        # Audio capture for Google Meet
        self.audio_capture = GoogleMeetAudioCapture()
        
        # Transcription handler
        self.transcription_handler = TranscriptionHandler()
        
        # Agentic RAG system for processing transcriptions
        self.rag_system = AgenticRAGSystem()
        
        # Track conversation history
        self.transcriptions = []
        self.questions = []
        self.answers = []
        self.strategies = []
        
        # Thread-safe list management
        self.list_lock = threading.Lock()

    def create_gradio_interface(self):
        """Create Gradio interface for Google Meet Companion"""
        with gr.Blocks() as demo:
            # Audio device selection
            device_dropdown = gr.Dropdown(
                label="Select Audio Input Device", 
                choices=[
                    f"{dev['index']}: {dev['name']}" 
                    for dev in self.audio_capture.list_audio_devices()
                ]
            )
            
            # Main interface components
            with gr.Row():
                with gr.Column():
                    transcription_output = gr.Textbox(label="Transcriptions")
                with gr.Column():
                    question_output = gr.Textbox(label="Identified Questions")
                    answer_output = gr.Textbox(label="Answers")
                    strategy_output = gr.Textbox(label="Retrieval Strategy")
            
            # Control buttons
            with gr.Row():
                start_btn = gr.Button("Start Recording")
                stop_btn = gr.Button("Stop Recording")
                save_btn = gr.Button("Save Recording")
                process_btn = gr.Button("Process Transcription")
            
            # Device selection interaction
            def start_recording(device_info):
                # Extract device index from dropdown
                device_index = int(device_info.split(':')[0]) if device_info else None
                self.audio_capture.start_recording(device_index)
            
            def stop_recording():
                self.audio_capture.stop_recording()
            
            def save_recording():
                return self.audio_capture.save_recorded_audio()
            
            def process_transcription():
                # Save current recording
                audio_file = self.audio_capture.save_recorded_audio()
                
                if audio_file:
                    # Transcribe saved audio
                    transcription = self.transcription_handler.transcribe_audio(audio_file)
                    
                    # Process transcription through Agentic RAG
                    if transcription:
                        rag_result = self.rag_system.process_transcription(transcription)
                        
                        # Thread-safe list update
                        with self.list_lock:
                            self.transcriptions.append(transcription)
                            self.questions.append(rag_result['question'])
                            self.answers.append(rag_result['answer'])
                            self.strategies.append(rag_result['strategy'])
                            
                            # Limit history
                            self.transcriptions = self.transcriptions[-10:]
                            self.questions = self.questions[-10:]
                            self.answers = self.answers[-10:]
                            self.strategies = self.strategies[-10:]
                        
                        return (
                            "\n".join(self.transcriptions), 
                            "\n".join(self.questions), 
                            "\n".join(self.answers),
                            "\n".join(self.strategies)
                        )
                
                return "", "", "", ""
            
            # Button click events
            start_btn.click(
                fn=start_recording, 
                inputs=[device_dropdown], 
                outputs=None
            )
            stop_btn.click(fn=stop_recording, inputs=None, outputs=None)
            save_btn.click(fn=save_recording, inputs=None, outputs=None)
            process_btn.click(
                fn=process_transcription, 
                inputs=None, 
                outputs=[transcription_output, question_output, answer_output, strategy_output]
            )

        demo.launch()

# Main execution
if __name__ == "__main__":
    # Required dependencies
    # print("Ensure you have installed: ")
    # print("- sounddevice")
    # print("- numpy")
    # print("- pyaudio")
    # print("- faster_whisper")
    # print("- gradio")
    # print("- google-generativeai")
    # print("- tavily-python")
    # print("- soundfile")
    
    # Initialize and launch the Google Meet Companion
    companion = GoogleMeetCompanion()
    companion.create_gradio_interface()
    companion = GoogleMeetCompanion()
    devices = companion.audio_capture.list_audio_devices()
    for device in devices:
        print(f"Device {device['index']}: {device['name']}")