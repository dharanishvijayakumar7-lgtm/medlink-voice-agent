import json
import logging
import textwrap
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    TurnHandlingOptions,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.plugins import ai_coustics
from rank_bm25 import BM25Okapi

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class MedLinkBM25RAG:
    """Fast BM25 retrieval for Indian OTC medicines"""

    def __init__(self, medicines_json: str):
        """Load medicines and initialize BM25"""
        if not Path(medicines_json).exists():
            logger.error(f"Medicines file not found: {medicines_json}")
            self.medicines = []
            self.bm25 = None
            return

        with open(medicines_json, encoding="utf-8") as f:
            self.medicines = json.load(f)

        logger.info(f"Loaded {len(self.medicines)} medicines for RAG")

        # Create searchable documents
        self.documents = []
        self.medicine_map = {}

        for idx, medicine in enumerate(self.medicines):
            name = medicine.get("name", "")
            composition = " ".join(medicine.get("composition", []))
            indications = " ".join(medicine.get("indications", []))
            therapeutic = medicine.get("therapeutic_class", "")

            # Weight indications 3x by repeating for better relevance
            doc = f"{name} {composition} {indications} {indications} {indications} {therapeutic}".lower()
            self.documents.append(doc)
            self.medicine_map[idx] = medicine

        # Initialize BM25
        tokenized_docs = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info("BM25 RAG initialized")

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Retrieve medicines for a query"""
        if not self.bm25:
            return []

        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        results = []
        for idx in top_indices:
            if idx < len(self.medicines):
                medicine = self.medicine_map[idx]
                results.append(
                    {
                        "name": medicine.get("name", ""),
                        "composition": medicine.get("composition", []),
                        "indications": medicine.get("indications", [])[:2],
                        "side_effects": medicine.get("side_effects", [])[:3],
                        "manufacturer": medicine.get("manufacturer", ""),
                        "therapeutic_class": medicine.get("therapeutic_class", ""),
                    }
                )

        return results


# Initialize RAG globally
rag = MedLinkBM25RAG("indian_otc_medicines_merged.json")


class MedLinkAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            llm=inference.LLM(model="google/gemma-4-31b-it"),
            instructions=textwrap.dedent(
                """\
                You are MedLink, your personal healthcare assistant. Your job is to understand what's troubling the patient, ask smart follow-up questions, reassure them, and guide them toward feeling better—all within this single conversation.
                
                You are speaking to people who may not be highly educated. Be PATIENT, CLEAR, and SIMPLE.
                
                # Your Opening
                Always start by warmly introducing yourself:
                "Hello, I'm MedLink, your healthcare assistant. I'm here to help you understand what's going on and guide you toward feeling better. What brings you in today?"
                
                # Speaking Style
                - Speak SLOWLY and clearly
                - Use simple words, not medical jargon
                - Repeat things if asked
                - Pronounce medicine names SLOWLY, one syllable at a time (e.g., "Para... ceta... mol")
                - Pause between sentences
                - Be patient and kind
                
                # Gathering Information
                Once the patient describes their main concern, ask smart follow-up questions ONE AT A TIME:
                - How long have you had this?
                - Where exactly does it hurt?
                - What makes it better or worse?
                - Any fever or other symptoms?
                - Have you had this before?
                
                Listen carefully to each answer before asking the next question.
                
                # Making Sense of It
                After you understand their symptoms, explain what might be causing it in simple language:
                "Based on what you're telling me, it sounds like you might be experiencing [condition]. This is very common."
                
                # Using the Medicine Tool
                Call the medicine_recommendations tool to find appropriate OTC medicines for their symptom.
                When recommending a medicine:
                1. Say the name slowly: "The medicine is called... [spell it out]"
                2. Explain what it does: "This medicine helps with..."
                3. How to take it: "You take it like this... [clear instructions]"
                4. Where to get it: "You can find this at any pharmacy or medical store"
                5. When to expect relief: "Most people feel better within..."
                
                # Emergency Override
                If patient mentions: chest pain, can't breathe, severe bleeding, unconsciousness, stroke signs, severe allergic reaction, thoughts of self-harm
                → IMMEDIATELY: "This is serious. Call emergency services (112) right away. Go to a hospital now."
                
                # Your Tone
                - Warm, caring, patient
                - Speak to them like a concerned friend
                - Simple, clear language
                - Never rush
                - Always show you're listening
                
                # Conversation Structure
                1. Warm introduction
                2. Listen to main complaint
                3. Ask 3-5 follow-up questions (wait for answers)
                4. Use medicine_recommendations tool
                5. Explain the medicine slowly
                6. Give practical self-care advice
                7. Reassure them
                8. Close warmly
                
                Remember: One conversation. One phone call. Make them feel better.
                """
            ),
        )

    @function_tool
    async def medicine_recommendations(self, context: RunContext, symptom: str):
        """Get OTC medicine recommendations for a symptom.

        Use this tool when the user has described their symptoms and you want to find
        appropriate over-the-counter medicines. Call this before recommending medicines.

        Args:
            symptom: The symptom or health concern (e.g., "fever", "headache", "cough")
        """
        logger.info(f"Retrieving medicines for symptom: {symptom}")

        results = rag.retrieve(symptom, top_k=3)

        if not results:
            return "I couldn't find specific medicines for that symptom right now. Let me suggest common options: rest, plenty of water, and see a doctor if it doesn't improve."

        # Format results for the LLM in patient-friendly way
        medicines_text = "Here are some over-the-counter medicines that might help:\n"
        for idx, med in enumerate(results, 1):
            comp = med["composition"][0] if med["composition"] else "Medicine"
            ind = med["indications"][0] if med["indications"] else "general wellness"

            medicines_text += f"\n{idx}. {med['name']} ({comp})\n"
            medicines_text += f"   What it does: Helps with {ind}\n"
            medicines_text += (
                "   Where to get it: Available at any pharmacy or medical store\n"
            )

        return medicines_text


server = AgentServer()


@server.rtc_session(agent_name="medlink-agent")
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=inference.STT(model="assemblyai/universal-3-5-pro", language="en"),
        tts=inference.TTS(
            model="fishaudio/s2.1-pro", voice="fa4c9eb3dccc4806b382b40d61c6b10a"
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            interruption={"mode": "adaptive"},
            preemptive_generation={"enabled": True},
        ),
        expressive=True,
    )

    await session.start(
        agent=MedLinkAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
