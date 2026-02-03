import { GoogleGenAI, Type, Chat, Modality } from "@google/genai";
import { Medication, DiaryEntry, SafetyAnalysis, FlareForecast, DietPlan, Symptom, Injection } from "../types";

const getAIClient = () => {
  return new GoogleGenAI({ apiKey: process.env.API_KEY });
};

let activeChatSession: Chat | null = null;

/**
 * Initializes the Doctor B chat session using the high-reasoning Pro model.
 */
export const startChat = (meds: Medication[], conditions: string[]) => {
  const ai = getAIClient();
  const medContext = meds.map(m => `${m.n} (${m.d}) at ${m.time}`).join(', ');
  const conditionContext = conditions.length > 0 
    ? `The patient's clinical background includes: ${conditions.join(', ')}.` 
    : "No chronic conditions listed.";
  
  activeChatSession = ai.chats.create({
    model: 'gemini-3-pro-preview',
    config: {
      systemInstruction: `You are Doctor B, an elite Senior Consultant Physician. 
      ${conditionContext}
      Current Regimen: ${medContext}. 
      Persona: Clinical, empathetic, concise, and professional. 
      Goal: Provide clear, evidence-based guidance on medication management and symptom tracking.`,
    },
  });
  return activeChatSession;
};

/**
 * Sends a message to the rewired Pro chat session.
 */
export const sendChatMessage = async (message: string) => {
  if (!activeChatSession) throw new Error("Consultation session not initialized");
  const response = await activeChatSession.sendMessage({ message });
  return response.text;
};

/**
 * Performs a deep clinical audit with reasoning (Thinking Budget).
 */
export const runClinicalAnalysis = async (
  meds: Medication[], 
  diary: DiaryEntry[], 
  symptoms: Symptom[], 
  conditions: string[]
): Promise<SafetyAnalysis> => {
  const ai = getAIClient();
  const prompt = `Perform a comprehensive safety audit for a patient with: ${conditions.join(', ')}.
    Current medications: ${meds.map(m => m.n + ' ' + m.d).join(', ')}.
    Recent symptoms: ${symptoms.map(s => `${s.name} (Severity: ${s.severity})`).join(', ')}.
    Patient logs: ${diary.map(d => d.text).join('; ')}.
    
    Identify potential drug interactions, side effects, and clinical correlations.`;

  const response = await ai.models.generateContent({
    model: 'gemini-3-pro-preview',
    contents: prompt,
    config: {
      systemInstruction: "You are a clinical safety auditor. Analyze interactions and patient data.",
      responseMimeType: "application/json",
      thinkingConfig: { thinkingBudget: 4000 },
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          interactions: { type: Type.ARRAY, items: { type: Type.STRING } },
          correlations: { type: Type.ARRAY, items: { type: Type.STRING } },
          summary: { type: Type.STRING }
        },
        required: ["interactions", "correlations", "summary"]
      }
    }
  });

  return JSON.parse(response.text || '{}');
};

/**
 * Generates natural audio for clinical reminders.
 */
export const generateReminderAudio = async (text: string): Promise<string | undefined> => {
  const ai = getAIClient();
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash-preview-tts",
    contents: { parts: [{ text: `Clinical Reminder: ${text}` }] },
    config: {
      responseModalities: [Modality.AUDIO],
      speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } } },
    },
  });
  return response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
};

/**
 * Advanced extraction for prescriptions (Supports Image & PDF).
 * Corrects and categorizes into AM (Morning) and PM (Night) schedules.
 */
export const analyzePrescription = async (base64Data: string, mimeType: string): Promise<{ meds: Medication[], injections: Injection[] }> => {
  const ai = getAIClient();
  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: {
      parts: [
        { text: "Analyze this document (Prescription/Report). Identify all medications. Categorize each into 'am' (Morning/Day) or 'pm' (Night/Evening) blocks based on frequency or specific instructions. Also identify injections (e.g. Humira, Enbrel, MTX). Extract dosages, frequencies, and administration sites. For weekly items, specify the day if mentioned." },
        { inlineData: { mimeType: mimeType, data: base64Data } }
      ]
    },
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          items: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                type: { type: Type.STRING, enum: ['pill', 'injection'] },
                name: { type: Type.STRING },
                dosage: { type: Type.STRING },
                time: { type: Type.STRING },
                frequency: { type: Type.STRING },
                site: { type: Type.STRING },
                schedule: { type: Type.ARRAY, items: { type: Type.STRING } },
                block: { type: Type.STRING, enum: ['am', 'pm'] }
              },
              required: ["type", "name", "dosage", "block"]
            }
          }
        }
      }
    }
  });

  const parsed = JSON.parse(response.text || '{"items":[]}');
  const meds: Medication[] = [];
  const injections: Injection[] = [];

  parsed.items.forEach((item: any) => {
    const commonDays = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    const finalSchedule = (item.schedule && item.schedule.length > 0) ? item.schedule : commonDays;

    if (item.type === 'injection') {
      injections.push({
        id: Math.floor(Math.random() * 1000000),
        name: item.name,
        dosage: item.dosage,
        site: item.site || 'Rotate Abdomen',
        time: item.time || (item.block === 'am' ? '08:00' : '20:00'),
        status: 'pending',
        frequency: item.frequency || 'Weekly',
        schedule: finalSchedule
      });
    } else {
      meds.push({
        id: Math.floor(Math.random() * 1000000),
        n: item.name,
        d: item.dosage,
        block: item.block || 'am',
        status: 'pending',
        time: item.time || (item.block === 'am' ? '08:00' : '20:00'),
        schedule: finalSchedule,
        count: 30
      });
    }
  });

  return { meds, injections };
};

/**
 * AI-powered clinical nutrition planning.
 */
export const generateDietPlan = async (meds: Medication[], conditions: string[]): Promise<DietPlan> => {
  const ai = getAIClient();
  const response = await ai.models.generateContent({
    model: 'gemini-3-pro-preview',
    contents: `Dietary optimization for patient with ${conditions.join(', ')}. Current meds: ${meds.map(m => m.n).join(', ')}.`,
    config: {
      responseMimeType: "application/json",
      thinkingConfig: { thinkingBudget: 2000 },
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          preferredFoods: { type: Type.ARRAY, items: { type: Type.STRING } },
          meals: { type: Type.ARRAY, items: { type: Type.OBJECT, properties: { name: { type: Type.STRING }, desc: { type: Type.STRING } } } },
          hydrationGoal: { type: Type.STRING }
        }
      }
    }
  });
  return JSON.parse(response.text || '{}');
};

/**
 * Computer vision pill identification (Supports Image & PDF).
 */
export const identifyPill = async (base64Data: string, mimeType: string): Promise<Partial<Medication>> => {
  const ai = getAIClient();
  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: {
      parts: [
        { text: "Identify this pill. Provide name, typical dosage, and clinical block (am or pm)." },
        { inlineData: { mimeType: mimeType, data: base64Data } }
      ]
    },
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: { 
          n: { type: Type.STRING }, 
          d: { type: Type.STRING }, 
          block: { type: Type.STRING, enum: ['am', 'pm'] } 
        },
        required: ["n", "d", "block"]
      }
    }
  });
  return JSON.parse(response.text || '{}');
};
