# SPDX-License-Identifier: Apache-2.0
"""
Hidden States Capture

Main interface for capturing hidden states from transformer models across different frameworks.
"""

from typing import List, Tuple, Union, Optional, Any
import torch
from ..adapters.base import LLMAdapter
from .storage import HiddenStatesStore, HiddenStatesCaptureContext
from .wrapper import wrap_transformer_layers
from ..detection.structure import get_optimal_layer_pattern


class HiddenStatesCapture:
    """
    Main class for capturing hidden states from transformer models
    
    This class provides a unified interface for capturing hidden states from different
    LLM frameworks through the adapter pattern.
    """
    
    def __init__(self, adapter: LLMAdapter, store: Optional[HiddenStatesStore] = None):
        """
        Initialize the capture system
        
        Args:
            adapter: LLMAdapter instance for the specific framework
            store: Optional HiddenStatesStore instance (creates new one if None)
        """
        self.adapter = adapter
        self.store = store if store is not None else HiddenStatesStore()
        self._wrapped_models = {}  # Cache wrapped models
    
    def wrap_model(self, llm: Any, layer_pattern: Optional[str] = None) -> None:
        """
        Wrap the model for hidden states capture
        
        Args:
            llm: The LLM instance
            layer_pattern: Optional layer pattern (auto-detected if None)
        """
        # Extract model
        model = self.adapter.extract_model(llm)
        model_id = id(model)
        
        # Check if already wrapped
        if model_id in self._wrapped_models:
            print('checked')
            return
        
        # Get layer pattern
        if layer_pattern is None:
            model_name = self.adapter.get_model_name(llm)
            layer_pattern = get_optimal_layer_pattern(model, model_name)
        
        # Wrap the model
        wrap_transformer_layers(model, layer_pattern, self.store)
        self._wrapped_models[model_id] = True
    
    def get_all_hidden_states(
        self, 
        llm: Any, 
        texts: List[str],
        split_by_samples: bool = True,
        auto_wrap: bool = True
    ) -> Union[Tuple[List[torch.Tensor], Any], Tuple[List[List[torch.Tensor]], Any]]:
        """
        Get all hidden states from the LLM
        
        Args:
            llm: The LLM instance
            texts: List of input texts
            split_by_samples: Whether to split hidden states by samples
            auto_wrap: Whether to automatically wrap the model
            
        Returns:
            If split_by_samples=True:
                (samples_hidden_states, outputs) where:
                - samples_hidden_states[sample_idx][layer_idx] is the hidden states
                  for sample_idx at layer_idx
                - outputs: LLM outputs
            If split_by_samples=False:
                (all_hidden_states, outputs) where:
                - all_hidden_states[layer_idx] is concatenated hidden states for layer_idx
                - outputs: LLM outputs
        """
        # Auto-wrap model if requested
        if auto_wrap:
            self.wrap_model(llm)
        
        # Determine if we need multi-batch mode
        multi_batch_mode = self.adapter.supports_multi_batch()
        
        # Setup capture context
        framework_context = self.adapter.setup_capture_context(llm)
        
        try:
            # Use context manager for capture
            with HiddenStatesCaptureContext(
                store=self.store, 
                multi_batch_mode=multi_batch_mode
            ) as capture:
                
                # Execute encoding
                outputs = self.adapter.encode_texts(llm, texts)
                
                # Get pooling metadata if available
                pooling_metadata = self.adapter.get_pooling_metadata(llm)
                if pooling_metadata is not None:
                    self.store.set_pooling_metadata(pooling_metadata)
                
                # Finalize multi-batch if needed
                if multi_batch_mode and not self.store.finalized:
                    self.store.finalize_multi_batch()
                
                # Get all hidden states
                all_hidden_states = capture.get_all_hidden_states()
                
                if split_by_samples:
                    # Split by samples
                    samples_hidden_states = self._split_hidden_states_by_samples(
                        all_hidden_states, outputs, llm
                    )
                    return samples_hidden_states, outputs
                else:
                    # Return concatenated hidden states
                    return all_hidden_states, outputs
                    
        finally:
            # Cleanup framework context
            if framework_context is not None:
                self.adapter.cleanup_capture_context(llm, framework_context)
    
    def _split_hidden_states_by_samples(
        self, 
        all_hidden_states: List[torch.Tensor],
        outputs: Any,
        llm: Any
    ) -> List[List[torch.Tensor]]:
        """
        Split hidden states by samples
        
        Args:
            all_hidden_states: All layer hidden states [layer_tensor, ...]
            outputs: LLM outputs
            llm: LLM instance
            
        Returns:
            samples_hidden_states[sample_idx][layer_idx]
        """
        if not all_hidden_states or not outputs:
            print("No hidden states or outputs to split.")
            print("Hidden states:", len(all_hidden_states))
            print("Outputs:",  len(outputs))
            return []
        
        # Try to get sample lengths from adapter
        sample_lengths = self.adapter.estimate_sample_lengths(llm, outputs)
        
        # If adapter can't estimate, try pooling metadata
        if sample_lengths is None:
            pooling_metadata = self.store.get_pooling_metadata()
            if pooling_metadata and hasattr(pooling_metadata, 'prompt_lens'):
                sample_lengths = pooling_metadata.prompt_lens
        
        # If still no sample lengths, fall back to uniform estimation
        if sample_lengths is None:
            if isinstance(outputs, list) and len(outputs) > 0:
                total_length = all_hidden_states[0].shape[0] if all_hidden_states else 0
                num_samples = len(outputs)
                if num_samples > 0:
                    avg_length = total_length // num_samples
                    sample_lengths = [avg_length] * num_samples
                    # Adjust last sample for remainder
                    if total_length % num_samples != 0:
                        sample_lengths[-1] += total_length % num_samples
                else:
                    return []
            else:
                return []
        
        # Check length matching
        expected_total_length = sum(sample_lengths)
        actual_length = all_hidden_states[0].shape[0] if all_hidden_states else 0
        
        if expected_total_length != actual_length:
            # Try to adjust sample lengths to match actual length
            if expected_total_length > actual_length:
                # Truncate samples
                adjusted_lengths = []
                current_length = 0
                for length in sample_lengths:
                    if current_length + length <= actual_length:
                        adjusted_lengths.append(length)
                        current_length += length
                    else:
                        remaining = actual_length - current_length
                        if remaining > 0:
                            adjusted_lengths.append(remaining)
                        break
                sample_lengths = adjusted_lengths
            else:
                # Expand last sample
                sample_lengths[-1] += actual_length - expected_total_length
        
        # Split hidden states by samples
        samples_hidden_states = []
        start_idx = 0
        
        for sample_length in sample_lengths:
            if sample_length <= 0:
                continue
                
            end_idx = start_idx + sample_length
            if end_idx > actual_length:
                end_idx = actual_length
            
            # Collect all layers for this sample
            sample_all_layers = []
            for layer_hidden_states in all_hidden_states:
                sample_layer_hidden_states = layer_hidden_states[start_idx:end_idx]
                sample_all_layers.append(sample_layer_hidden_states)
            
            samples_hidden_states.append(sample_all_layers)
            start_idx = end_idx
            
            if start_idx >= actual_length:
                break
        
        return samples_hidden_states
    
    def clear(self):
        """Clear all stored hidden states"""
        self.store.clear()
    
    def get_layer_info(self):
        """Get information about captured layers"""
        return self.store.get_layer_info()
    
    def get_layer_count(self):
        """Get number of captured layers"""
        return self.store.get_layer_count() 