# Demo conversation report ? 13 November 2026

**Finding: the brief English hair-loss conversation is substantially better, but question relevance and language consistency still need work. The current results do not establish reliable behavior across all six languages.**

## Method and results

Ran 16 synthetic text conversations through the current ConversationAgent using the real configured Sarvam LLM. These were not telephone calls: microphone input, STT, TTS, background noise and handset playback were not tested. Caller facts were fixed; regional-language caller messages were translated through Sarvam. Database and Firestore writes were disabled.

- 13 conversations completed: 12 routine assessments and one emergency response.
- Completed routine conversations averaged **4.0 follow-up questions**; every one asked four. The emergency response asked zero routine questions before urgent instructions.
- All 12 completed routine assessments contained possible-cause/uncertainty, explanation, do?s, don?ts and referral sections. This checks structure, not diagnostic accuracy or language quality.
- Three regional-language attempts stopped with APIConnectionError. These are included as failures, not excluded or counted as successful demos.
- The regression suite passed 519 tests. The conversational weaknesses below were found by reviewing actual replies despite passing structural checks.

## Exact English follow-up questions

### Gradual hair loss for two years

1. How long have you been noticing the hair loss?
2. Where is the thinning: all over, in separate patches, or mainly along the hairline?
3. Have you noticed any changes to your scalp, such as itching or flaking?
4. Does anyone in your family have a similar pattern of hair thinning?

The caller answered ?About two years.? The agent continued with pattern, scalp and family-history questions. Its closing described hereditary-pattern thinning as a possibility, connected that to the reported front-hairline pattern and family history, and included hair-care do?s, don?ts and referral guidance.

### Hair loss in separate patches

1. How long have you been noticing the hair loss?
2. Have you noticed any changes to your scalp, such as itching or flaking?
3. Does anyone in your family have a similar pattern of hair thinning?
4. Was there an illness or major change in your health before the shedding began?

The closing acknowledged several possible causes and recommended examination alongside practical hair-care guidance. It did not confidently assign a specific diagnosis.

### Hair loss with an itchy, flaky scalp

1. How long have you been noticing the hair loss?
2. Where is the thinning: all over, in separate patches, or mainly along the hairline?
3. Have you noticed any changes to your scalp, such as itching or flaking?
4. Does anyone in your family have a similar pattern of hair thinning?

The agent suggested a scalp condition as a possible contributor. However, the opening had already mentioned itching, so the scalp question partly repeated supplied information rather than asking a more precise missing detail.

### Detailed opening with duration, pattern and family history

1. Has the thinning been gradual, or has it changed suddenly?
2. Was there an illness or major change in your health before the shedding began?
3. How do you usually style or treat your hair?
4. How is the hair loss affecting your daily life?

**Quality failure:** the caller had already explicitly said ?gradually.? Question 1 re-asked that fact. The other questions explored additional context.

### Persistent hiccups (no matched reference)

1. How long have you had these hiccups?
2. How severe is it right now?
3. Are there any other symptoms?
4. How old is the person who is unwell?

**Coverage limitation:** these questions were generic. The closing admitted uncertainty and offered general precautions and review advice, without a symptom-specific explanation.

### Crushing chest pain and breathing difficulty

No routine follow-ups. The first response gave emergency guidance, followed by a question about whether someone nearby could help.

## Regional-language results

| Intended language | Scenario | Routine questions | Observation |
|---|---|---:|---|
| Hindi | gradual | 4 | Completed, but possible cause and explanation were in English; care guidance was Hindi. |
| Hindi | patchy | 4 | Completed with Hindi guidance; used generic severity before hair-pattern and scalp questions. |
| Tamil | gradual | 4 | Caller-translation harness drifted into Malayalam; this is not a valid Tamil-only result. |
| Tamil | patchy | 4 | Caller-translation harness drifted into Malayalam; this is not a valid Tamil-only result. |
| Telugu | gradual | 4 | Completed, but possible cause and explanation were in English; care guidance was Telugu. |
| Telugu | patchy | 4 | Completed, but possible cause and explanation were in English; care guidance was Telugu. |
| Kannada | gradual | 4 | Completed, but possible cause and explanation were in English; care guidance was Kannada. |
| Kannada | patchy | Incomplete | API failure; the agent also emitted its connection-problem message. No completed assessment. |
| Malayalam | gradual | Incomplete | API failure during the simulated conversation; no completed assessment. |
| Malayalam | patchy | Incomplete | API failure translating the opening before the agent could be exercised. |

For completed gradual-hair-loss cases, the usual question topics were duration, hair-loss pattern, scalp changes and family history. Hindi asked scalp changes before pattern. Exact native-script wording and full responses are preserved in the linked transcripts below.

## What should change next

1. **Validate the actual spoken language of every assessment field.** A correct language code in JSON did not stop English explanation text from reaching a regional-language conversation.
2. **Improve recognition of volunteered facts.** ?Gradually? and scalp itching should suppress equivalent follow-ups or produce a more specific question. A four-question minimum should not be filled with repeats.
3. **Broaden symptom-specific coverage.** The unsupported hiccup scenario demonstrated that the generic fallback still reduces question quality.
4. **Make the test caller reliable.** Use reviewed fixed translations and reject mismatched scripts before sending them to the agent; the current Tamil trials cannot establish Tamil performance.
5. **Investigate API failures separately from phone/network quality.** The failure set includes caller-translation requests; it does not establish that the user?s regional network caused them. Keep failures visible and avoid treating them as medical conclusions.

The hair-loss closing advice was consistently present in completed cases. Clinical accuracy has not been established by clinician review, and a four-question conversation cannot confirm a diagnosis. Real phone playback and interruption behavior still require a handset test.

## Evidence

- [Six English transcripts, fixed facts and structured responses](demo_calls_english_final.json)
- [Ten regional-language attempts, including failures](demo_calls_languages_final.json)
- [Reproducible audit harness](../scripts/audit_question_quality.py)

The averages above exclude incomplete attempts and the emergency case. The report retains failed attempts and language problems explicitly; a structural pass is not an overall quality pass.
