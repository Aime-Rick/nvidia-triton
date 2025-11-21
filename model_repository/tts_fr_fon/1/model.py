import json
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import pipeline
import torch


class TritonPythonModel:
    """
    Modèle TTS pour traduire du français vers le fon et générer l'audio.
    """
    
    def initialize(self):
        """
        Initialise les modèles de traduction et de synthèse vocale.
        """
        # Récupérer la configuration du modèle
        # self.model_config = json.loads(args['model_config'])
        
        # Déterminer le device (GPU si disponible)
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Logger
        self.logger = pb_utils.Logger
        self.logger.log_info(f"Initializing TTS model on device: {'GPU' if self.device >= 0 else 'CPU'}")
        
        try:
            # Initialiser le pipeline de traduction français -> fon
            self.logger.log_info("Loading translation model (NLLB-200)...")
            self.translator = pipeline(
                "translation",
                model="facebook/nllb-200-distilled-600M",  # Version plus légère
                src_lang="fra_Latn",
                tgt_lang="fon_Latn",
                device=self.device
            )
            
            # Initialiser le pipeline TTS pour le fon
            self.logger.log_info("Loading TTS model (MMS-TTS-FON)...")
            self.tts = pipeline(
                "text-to-speech",
                model="facebook/mms-tts-fon",
                device=self.device
            )
            
            self.logger.log_info("Models loaded successfully!")
            
        except Exception as e:
            self.logger.log_error(f"Error during initialization: {str(e)}")
            raise pb_utils.TritonModelException(f"Failed to initialize models: {str(e)}")
        
    def execute(self, requests):
        """
        Traite les requêtes d'inférence.
        
        Args:
            requests: Liste de pb_utils.InferenceRequest
            
        Returns:
            Liste de pb_utils.InferenceResponse
        """
        responses = []
        
        for request in requests:
            try:
                # Récupérer le texte d'entrée
                input_tensor = pb_utils.get_input_tensor_by_name(request, "text")
                input_text = input_tensor.as_numpy()[0].decode('utf-8')
                
                self.logger.log_info(f"Processing text: {input_text[:50]}...")
                
                # Étape 1: Traduire du français vers le fon
                self.logger.log_verbose("Translating French to Fon...")
                translated_result = self.translator(input_text)
                translated_text = translated_result[0]['translation_text']
                self.logger.log_info(f"Translated text: {translated_text[:50]}...")
                
                # Étape 2: Générer l'audio à partir du texte fon
                self.logger.log_verbose("Generating speech from Fon text...")
                audio_result = self.tts(translated_text)
                
                # Extraire l'audio et le taux d'échantillonnage
                audio_array = audio_result["audio"]
                sampling_rate = audio_result["sampling_rate"]
                
                # Convertir en numpy arrays avec les bons types
                audio_np = np.array(audio_array, dtype=np.float32).flatten()
                sampling_rate_np = np.array([sampling_rate], dtype=np.int32)
                
                self.logger.log_info(f"Generated audio: {len(audio_np)} samples at {sampling_rate} Hz")
                
                # Créer les tenseurs de sortie
                audio_tensor = pb_utils.Tensor("audio", audio_np)
                sampling_rate_tensor = pb_utils.Tensor("sampling_rate", sampling_rate_np)
                translated_text_tensor = pb_utils.Tensor(
                    "translated_text", 
                    np.array([translated_text.encode('utf-8')], dtype=object)
                )
                
                # Créer la réponse
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[audio_tensor, sampling_rate_tensor, translated_text_tensor]
                )
                
                responses.append(inference_response)
                
            except Exception as e:
                self.logger.log_error(f"Error processing request: {str(e)}")
                # Créer une réponse d'erreur
                error_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"Error during inference: {str(e)}")
                )
                responses.append(error_response)
        
        return responses
    
    def finalize(self):
        """
        Nettoie les ressources avant de décharger le modèle.
        """
        self.logger.log_info("Finalizing TTS model...")
        # Libérer les ressources
        self.translator = None
        self.tts = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.log_info("Model finalized successfully.")



   
