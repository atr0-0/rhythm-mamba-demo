'use client';

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  ReactNode,
} from 'react';
import { VitalsState, initialVitalsState, InferenceResponse } from '@/lib/types';
import { runInference as runInferenceApi, validateVideoFile } from '@/lib/api';

interface VitalsContextType {
  vitals: VitalsState;
  runInference: (file?: File) => Promise<void>;
  selectFile: (file: File) => void;
  setModelChoice: (choice: 'prebuilt' | 'self') => void;
  clearVitals: () => void;
  setError: (error: string | null) => void;
  setLoading: (loading: boolean) => void;
  showResultsModal: boolean;
  openResultsModal: () => void;
  closeResultsModal: () => void;
}

const VitalsContext = createContext<VitalsContextType | undefined>(undefined);

export function VitalsProvider({ children }: { children: ReactNode }) {
  const [vitals, setVitals] = useState<VitalsState>(initialVitalsState);
  const [showResultsModal, setShowResultsModal] = useState(false);

  const handleRunInference = useCallback(
    async (fileOverride?: File) => {
      const file = fileOverride || vitals.uploadedFile;
      if (!file) {
        setVitals((prev) => ({
          ...prev,
          error: 'Please upload a video first',
        }));
        return;
      }

      // Validate file
      const validation = validateVideoFile(file);
      if (!validation.valid) {
        setVitals((prev) => ({
          ...prev,
          error: validation.error || 'Invalid file',
          isLoading: false,
        }));
        return;
      }

      // Ensure preview exists
      if (!vitals.videoPreviewUrl) {
        const previewUrl = URL.createObjectURL(file);
        setVitals((prev) => ({
          ...prev,
          videoPreviewUrl: previewUrl,
        }));
      }

      // Set loading state
      setVitals((prev) => ({
        ...prev,
        isLoading: true,
        error: null,
      }));

      try {
        // Run inference
        const result = await runInferenceApi(file, vitals.modelChoice, (message) => {
          console.log('Inference progress:', message);
        });

        // Update vitals with results
        setVitals((prev) => ({
          ...prev,
          hr: result.hr,
          hrConf: result.hr_conf,
          snr: result.snr,
          quality: result.quality,
          waveform: result.waveform,
          processingTime: result.processing_time,
          fps: result.fps,
          isLoading: false,
          error: null,
        }));
        
        // Open results modal after successful inference
        setShowResultsModal(true);
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : 'Inference failed';
        setVitals((prev) => ({
          ...prev,
          isLoading: false,
          error: errorMessage,
        }));
      }
    },
    [vitals.modelChoice, vitals.uploadedFile, vitals.videoPreviewUrl]
  );

  const handleSelectFile = useCallback((file: File) => {
    const validation = validateVideoFile(file);
    if (!validation.valid) {
      setVitals((prev) => ({
        ...prev,
        error: validation.error || 'Invalid file',
      }));
      return;
    }

    if (vitals.videoPreviewUrl) {
      URL.revokeObjectURL(vitals.videoPreviewUrl);
    }

    const previewUrl = URL.createObjectURL(file);
    setVitals((prev) => ({
      ...initialVitalsState,
      modelChoice: prev.modelChoice,
      uploadedFile: file,
      videoPreviewUrl: previewUrl,
    }));
  }, [vitals.videoPreviewUrl]);

  const handleClearVitals = useCallback(() => {
    // Cleanup preview URL
    if (vitals.videoPreviewUrl) {
      URL.revokeObjectURL(vitals.videoPreviewUrl);
    }
    setVitals((prev) => ({
      ...initialVitalsState,
      modelChoice: prev.modelChoice,
    }));
  }, [vitals.videoPreviewUrl]);

  const handleSetError = useCallback((error: string | null) => {
    setVitals((prev) => ({
      ...prev,
      error,
    }));
  }, []);

  const handleSetLoading = useCallback((loading: boolean) => {
    setVitals((prev) => ({
      ...prev,
      isLoading: loading,
    }));
  }, []);

  const handleSetModelChoice = useCallback((choice: 'prebuilt' | 'self') => {
    setVitals((prev) => ({
      ...prev,
      modelChoice: choice,
    }));
  }, []);

  const handleOpenResultsModal = useCallback(() => {
    setShowResultsModal(true);
  }, []);

  const handleCloseResultsModal = useCallback(() => {
    setShowResultsModal(false);
  }, []);

  const value: VitalsContextType = {
    vitals,
    runInference: handleRunInference,
    selectFile: handleSelectFile,
    setModelChoice: handleSetModelChoice,
    clearVitals: handleClearVitals,
    setError: handleSetError,
    setLoading: handleSetLoading,
    showResultsModal,
    openResultsModal: handleOpenResultsModal,
    closeResultsModal: handleCloseResultsModal,
  };

  return (
    <VitalsContext.Provider value={value}>{children}</VitalsContext.Provider>
  );
}

export function useVitals(): VitalsContextType {
  const context = useContext(VitalsContext);
  if (context === undefined) {
    throw new Error('useVitals must be used within a VitalsProvider');
  }
  return context;
}
