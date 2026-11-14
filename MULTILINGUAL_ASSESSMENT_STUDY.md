# MedLink multilingual assessment and speech study

14 November 2026. User-reported issue: partly spoken words and inaccurate-sounding Telugu assessment. The exact symptom, wording and call recording were not available, so this study does not claim to reproduce that particular call.

## Findings and changes

The strongest reproducible defects were in assessment validation, reference selection and recovery behavior. The audio checks did not establish a single cause of cut-off words on a real handset.

| Finding | Effect | Change |
|---|---|---|
| A response's language code was checked, but individual spoken fields were not | English explanations or mixed scripts could reach a regional-language TTS voice | Check each spoken assessment field, question and follow-up against the selected script; reject ordinary English diagnosis words embedded in native speech |
| Complete sentences were not required | A visibly unfinished field could be accepted as speech | Reject missing terminal punctuation, ellipses, replacement characters and corrupt control characters in assessment speech; never guess the missing ending |
| `likely` and `avoid` are stored but not spoken; the model often omitted them | Useful spoken explanations were rejected and replaced with generic recovery text | Derive missing internal cause text from the actual explanation and use a conservative localized avoidance default; still validate all content |
| Generated explanations had no dedicated evidence field | A plausible cause could be stated without a traceable caller detail | Require an exact original-language `explanation_quote` before accepting a generated assessment; strengthen instructions about uncertainty and unsupported causes |
| Long native-language inputs could select an unrelated reference, including from denied or incidental symptoms | Telugu hair thinning selected skin itching; unsupported hiccups could match breathing difficulty or other unrelated entries | Disable BM25 fallback on the active conversation path, require a short English primary complaint, and confirm/replace the provisional reference before committing the assessment |
| Runtime state was a separate system message after the caller | Model proposals frequently ignored question count and validation corrections in these fixtures | Consolidate instructions, state and schema into one leading system message; explicitly require assessment at the question ceiling |
| Missing regional-language recovery guidance returned no assessment | Rejected model output could produce a connection-problem message even without an API outage | Use honest localized uncertainty and general next steps when a known reference lacks localized recovery; actual API errors remain service errors |
| English medicine-name checks missed native-script names | A Telugu fever proposal included transliterated ORS despite the active medicine-free policy | Add supplemental native-script medicine-name checks and omit medicine-bearing self-care sentences from the model reference |
| One voice and default synthesis settings were used throughout | Language-specific pronunciation and clause pauses were not explicitly tuned | Configure Neha for Telugu, retain Ritu elsewhere, expose per-language overrides, and set pace 0.95, temperature 0.4, buffer 100 and maximum chunk length 500 |
| Provider sentence tokenization recognizes periods but not Hindi danda | Native punctuation could leave longer text units for synthesis | Normalize danda and semicolons to full stops at the active TTS boundary without slicing native words |

Script checks are heuristics, not language identification or translation certification. Native-script jargon, incorrect grammar and clinically inappropriate statements can still pass. An exact quote proves that the quoted text occurred; it does not prove that the proposed cause follows medically from it.

## What diagnoses does the agent actually say?

There is no fixed list of diagnoses in the active agent. Sarvam generates a tentative explanation using the conversation and a retrieved reference. The five internal assessment fields are `likely`, `why`, `do`, `avoid` and `seek_care`. The caller hears **`why` + `do` + `seek_care`**. Reading only `likely` from the records does not tell us what the caller heard.

The current reference library contains these 23 presentations:

| Reference group | Presentations |
|---|---|
| General and respiratory | Fever; cough/common cold; sore throat; breathing difficulty; body/muscle/joint aches; weakness/tiredness |
| Digestive | Diarrhoea; vomiting; abdominal pain; acidity/heartburn/indigestion; intestinal worms; constipation |
| Skin and hair | Hair loss/thinning; itchy skin/rash; fungal skin/ringworm; cuts/wounds; burns/scalds |
| Other | Headache; chest pain/discomfort; ear pain/discharge; urinary symptoms; allergic sneezing/itchy eyes; pregnancy-related concerns |

These are reference topics, not 23 conditions that the agent can reliably diagnose. Some labels themselves name a condition, but matching the reference must not be presented as proof of it. Complaints outside this set should remain explicitly uncertain.

### Explanations reviewed in the synthetic study

| Scenario | Observed explanation type | Appropriate interpretation |
|---|---|---|
| Gradual front-hairline thinning and similar paternal history | Inherited-pattern thinning as a possibility, or uncertainty about cause | The history can support that possibility; examination is still needed to establish the cause |
| Mild recent fever and body aches | Possible infection, or inability to determine a specific cause | These symptoms alone cannot establish a viral/bacterial cause or a particular infection |
| Gradual headache after extended screen use | Possible eye strain/tension-related explanation, or uncertainty | A temporal association can suggest a possibility; it does not confirm a diagnosis or rule out serious causes |
| Intermittent hiccups, outside the library | Uncertain cause and medical review advice | Do not invent reflux or breathing difficulty from incidental words or retrieval noise |

The medical distinction matters: a hair-loss assessment can require examination and sometimes testing, and headaches have multiple possible causes and care warnings. The software changes do not replace that evaluation. See [Mayo Clinic: hair-loss diagnosis](https://www.mayoclinic.org/diseases-conditions/hair-loss/diagnosis-treatment/drc-20372932), [NHS: headaches](https://www.nhs.uk/symptoms/headaches/) and [NHS: fever in adults](https://www.nhs.uk/symptoms/fever-in-adults/).

## How the study was run

1. Reviewed existing source, saved conversation audits and saved audio results.
2. Ran the original regression suite: **530 passed**.
3. Added tests before the corresponding behavior changes. The new tests reproduced wrong-language output, unfinished fields, missing metadata, runtime instruction placement and native-script medicine bypasses.
4. Ran a controlled assessment-turn fixture with fixed synthetic native-language inputs for hair thinning, fever, headache and unsupported hiccups: four scenarios in each of six languages.
5. Repeated that fixture with the committed original conversation implementation and then the changed implementation. The original source used for comparison is saved in `assessment_audit_out/baseline_conversation.py`.
6. Ran six complete synthetic hair-loss text conversations using fixed native-language caller answers and the real configured Sarvam model. No model translated or improvised the callers.
7. Synthesized selected assessment closings, levelled the audio, resampled it to 8 kHz, saved WAVs, and transcribed the samples back through Sarvam STT.
8. Ran the final complete regression suite: **598 passed**. Ruff checks and formatting passed for the changed code; `git diff --check` passed.

The assessment-turn fixture sets six questions in application state to exercise the final-turn path. It is not evidence that six real questions were asked. Only the separate full-conversation fixture measures a complete question flow. Inputs and translations remain drafts awaiting human language review. These are synthetic data; database and Firestore writes were disabled in the audit scripts.

## Assessment results by language

| Language | Original: completed assessment turns / 4 | Changed: completed assessment turns / 4 | Separate full hair conversation |
|---|---:|---:|---|
| English | 4/4 | 4/4 | Completed after four questions |
| Hindi | 1/4 | 4/4 | Completed after four questions |
| Tamil | 1/4 | 4/4 | Completed after four questions |
| Telugu | 0/4 | 4/4 | Completed after four questions |
| Kannada | 0/4 | 4/4 | Completed after four questions |
| Malayalam | 1/4 | 4/4 | Completed after four questions |

The controlled fixture improved from **7/24 to 24/24 completed assessments**. This measures completion under these inputs, not clinical accuracy or a general reliability rate. Of the final 24 responses, 14 retained explanation text appearing in the model proposals; the other 10 used recovery explanations. Recovery is counted explicitly: generic uncertainty must not be presented as successful individualized diagnosis.

In the final verification, the three covered scenarios selected their expected reference across all six languages, and unsupported hiccups retained no matched reference. The intermediate snapshots show why this correction was necessary, including an earlier Kannada fever case selecting breathing difficulty.

The full hair conversations produced the following explanation sentences:

| Language | Actual recorded explanation |
|---|---|
| English | “This may be related to hereditary pattern hair loss, since you mentioned your father has similar thinning.” |
| Hindi | “हो सकता है कि यह खानदानी वजह से हो, क्योंकि आपने बताया कि आपके पिता के बाल भी इसी तरह कम हुए हैं।” |
| Tamil | “இது பரம்பரையாக வரக்கூடிய முடி மெலிதல் காரணமாக இருக்கலாம், ஏனெனில் உங்கள் அப்பாவுக்கும் இதேபோல் முடி குறைந்துள்ளது என்று சொன்னீர்கள்.” |
| Telugu | “మీ నాన్నగారికి కూడా ఇలాగే జుట్టు తగ్గింది కాబట్టి, ఇది వంశపారంపర్యంగా వచ్చే జుట్టు పలచబడటం కావచ్చు.” |
| Kannada | “ನಿಮ್ಮ ತಂದೆಯವರಿಗೂ ಇದೇ ರೀತಿ ಕೂದಲು ತೆಳುವಾಗಿದ್ದರಿಂದ, ಇದು ವಂಶಪಾರಂಪರ್ಯವಾಗಿರಬಹುದು.” |
| Malayalam | “മുടി കൊഴിച്ചിലിന് പല കാരണങ്ങളുണ്ടാകാം; പരിശോധനയില്ലാതെ നിങ്ങളുടെ കാരണം ഉറപ്പിക്കാനാവില്ല.” |

Malayalam used the more general recovery explanation in that full conversation. The examples above show actual output, not native-speaker approval. The complete transcripts include the practical care and referral sentences as well.

## Half-spoken words: what was and was not established

There are separate failure paths:

- **Incomplete generated text:** now rejected in assessment fields instead of being passed to TTS.
- **Pronunciation and chunk boundaries:** punctuation normalization, larger synthesis context and delivery tuning address plausible contributors. They are not proof of a handset fix.
- **Interrupted playback:** a real caller, background voice or echo can truncate active speech when interruptions are enabled. The local demo configuration previously used protected replies. This change does not silently change the interruption setting.
- **Provider, network or phone playback:** still needs a real call with the running worker and handset. A local WAV cannot test cellular transport or the user's speaker.

LiveKit documents that interruptions truncate the segment in progress. Sarvam documents voice choices by language and the effects of pace and temperature; its guide recommends Neha/Priya for Telugu. These informed the tuning, but do not establish that one voice is perceptually best. See [LiveKit pipeline nodes](https://docs.livekit.io/agents/logic/nodes/) and [Sarvam Bulbul v3 guidance](https://docs.sarvam.ai/api/api-guides-tutorials/text-to-speech/bulbul-v3-best-practices).

### Saved audio checks

One hair-assessment closing per language was synthesized using the tuned settings and configured voice. The input texts are saved in the result JSON and derive from the `final.json` assessment snapshot.

| Language | Voice | Audio length | Character difference after STT | Longest internal quiet interval |
|---|---|---:|---:|---:|
| English | Ritu | 19.21 s | 4.27% | 240 ms |
| Hindi | Ritu | 19.24 s | 6.10% | 360 ms |
| Tamil | Ritu | 23.98 s | 0.71% | 480 ms |
| Telugu | Neha | 34.80 s | 0.75% | 610 ms |
| Kannada | Ritu | 22.94 s | 1.53% | 770 ms |
| Malayalam | Ritu | 25.18 s | 5.63% | 1090 ms |

Character difference is a spacing-insensitive sequence comparison, not word accuracy or comprehension. STT and TTS from the same vendor can share biases. Quiet intervals can be natural phrasing; the Kannada/Malayalam values warrant listening, not an automatic conclusion that audio is broken. Low tail energy does not prove that the final word is intact.

A separate Telugu comparison used the same assessment and tuned parameters:

| Voice | Duration | Character difference | Longest quiet interval |
|---|---:|---:|---:|
| Ritu | 27.17 s | 1.14% | 760 ms |
| Neha | 33.25 s | 3.03% | 550 ms |

Neha had a shorter longest pause but slower delivery and a higher transcription difference in this single comparison. There is **no demonstrated overall winner**. Both WAVs are provided for a Telugu speaker to compare. An earlier `telugu_voice_comparison` folder contains a service-error message and must not be used as assessment-pronunciation evidence.

## Files to review and listen to

- [Final assessment fixtures and model proposals](assessment_audit_out/verification.json)
- [Original implementation comparison](assessment_audit_out/baseline_six.json)
- [Six full text conversations](assessment_audit_out/final_conversations.json)
- [Summary counts](assessment_audit_out/summary.json)
- [Audio metrics and transcriptions](assessment_audit_out/audio_verified/results.json)
- [Telugu Neha sample](assessment_audit_out/audio_verified/te-IN_neha.wav)
- [Telugu comparison: Ritu](assessment_audit_out/telugu_assessment_comparison/te-IN_ritu.wav)
- [Telugu comparison: Neha](assessment_audit_out/telugu_assessment_comparison/te-IN_neha.wav)
- [English](assessment_audit_out/audio_verified/en-IN_ritu.wav), [Hindi](assessment_audit_out/audio_verified/hi-IN_ritu.wav), [Tamil](assessment_audit_out/audio_verified/ta-IN_ritu.wav), [Kannada](assessment_audit_out/audio_verified/kn-IN_ritu.wav), [Malayalam](assessment_audit_out/audio_verified/ml-IN_ritu.wav)

Intermediate files (`baseline`, `improved`, `improved_final`, `final`, `prompt_probe`) document investigations at different stages. They should not be combined into one performance score.

## Configuration and use

The models remain unchanged: Sarvam realtime STT, `sarvam-105b-conversations` and `bulbul:v3`. The changes are application validation, reference handling and synthesis settings.

New optional environment settings are documented in `.env.example`:

```dotenv
MEDLINK_TTS_LANGUAGE_SPEAKERS={"te-IN":"neha"}
MEDLINK_TTS_PACE=0.95
MEDLINK_TTS_TEMPERATURE=0.4
MEDLINK_TTS_MIN_BUFFER_SIZE=100
MEDLINK_TTS_MAX_CHUNK_LENGTH=500
```

Use `{}` for the language map to use the globally configured voice everywhere. The global default remains Ritu. Restart the intended worker to load these source changes:

```powershell
uv run src/agent.py start
```

This study did not restart or redeploy the running worker. Avoid starting another worker with the same agent name while an older one remains online. Then verify a short Telugu call and one call in each other language. Record the actual output and any interrupted-speech events; compare the handset with the saved WAVs.

## Remaining limits and next review

The reference library still has condition-specific localized recovery guidance only for hair loss. Other regional-language failures use generic uncertainty, so coverage needs reviewed expansion. The finite medicine list does not recognize every spelling or compound. Script checks do not certify the language of native-script transliterations or the meaning of an explanation. One sample per scenario is too small to estimate reliability.

A clinician should review the complete explanation, care advice and escalation criteria. A native speaker of each language should review naturalness, terminology, negation and whether words sound complete on the saved audio and the real phone route. In particular, listen to the longer Malayalam/Kannada pauses and the Telugu speed/voice tradeoff.

No claim is made that clinical diagnostic accuracy has been measured, that all speech truncation is solved, or that the project is ready for unattended patient use.

### Credential exposure during testing

A newly written failing test attempted to patch a configuration field before that field existed; pytest printed the settings object's representation, which included credential values. Sensitive fields now have `repr=False`, and a regression verifies they are absent from the representation. This does not erase the earlier tool output. Rotate exposed provider credentials. Phone-hashing and field-encryption keys require a planned data migration rather than blind replacement, to preserve record lookup and decryption. No credentials were changed by this task.
