-- ============================================================
-- LiftMind KB migration: Supermec ERROR 4 backfill
-- Run once on the live PostgreSQL database.
-- Generated: 2026-04-17
-- ============================================================

BEGIN;

-- ── New facts ────────────────────────────────────────────────
INSERT INTO facts (document_id, lift_models, content, keywords, fact_type, confidence_score, drive_types, door_types)
VALUES (NULL, '{"Supermec"}', 'ERROR 4 on the Supermec SED2 drive card indicates a tracking error: the actual motor speed (from the encoder) deviates from the theoretical speed by more than the SGL_ERR threshold (Parameter 74). The primary cause is encoder malfunction — the encoder may be disconnected, have a broken cable, or have failed internally.', '{"ERROR 4","error4","tracking error","encoder","SED2","drive card","SGL_ERR"}', 'diagnostic', 0.9, '{"hydraulic"}', '{}')
ON CONFLICT DO NOTHING;
INSERT INTO facts (document_id, lift_models, content, keywords, fact_type, confidence_score, drive_types, door_types)
VALUES (NULL, '{"Supermec"}', 'To diagnose ERROR 4 on Supermec: check the encoder cable connection at J3 (shunt box) and J8 (activation board). Inspect the shielded cable for damage or broken shielding. Verify encoder supply voltage. If cable is intact, the encoder itself may have failed and requires replacement.', '{"ERROR 4","error4","encoder","J3","J8","shunt box","activation board","diagnostic"}', 'procedure', 0.9, '{"hydraulic"}', '{}')
ON CONFLICT DO NOTHING;
INSERT INTO facts (document_id, lift_models, content, keywords, fact_type, confidence_score, drive_types, door_types)
VALUES (NULL, '{"Supermec"}', 'ERROR 4 tracking error threshold is set by SGL_ERR (Parameter 74), expressed as maximum admissible error in (rev/min)×4 between theoretical and real speed. If SGL_ERR is set too tight it can cause nuisance ERROR 4 trips even with a healthy encoder.', '{"ERROR 4","error4","SGL_ERR","parameter 74","tracking error","threshold","rev/min"}', 'setting', 0.9, '{"hydraulic"}', '{}')
ON CONFLICT DO NOTHING;
INSERT INTO facts (document_id, lift_models, content, keywords, fact_type, confidence_score, drive_types, door_types)
VALUES (NULL, '{"Supermec"}', 'When Supermec shows ERROR 4: CP relay and RE relay remaining energised (continuously supplied) does NOT rule out encoder failure — the relays reflect activation board battery status, not drive card encoder health. The red LED on the keypad confirms ERROR 4 is active on the SED2 drive card.', '{"ERROR 4","error4","CP relay","RE relay","red LED","activation board","SED2","encoder"}', 'diagnostic', 0.9, '{"hydraulic"}', '{}')
ON CONFLICT DO NOTHING;
INSERT INTO facts (document_id, lift_models, content, keywords, fact_type, confidence_score, drive_types, door_types)
VALUES (NULL, '{"Supermec"}', 'EPCHOPPER E1–E5 codes are separate from ERROR 1/2/4: EPCHOPPER errors relate to the battery charger (CN1 circuit), not to the SED2 drive card tracking logic.', '{"EPCHOPPER","E1","E2","E3","E4","E5","ERROR 4","battery charger","CN1","SED2"}', 'diagnostic', 0.9, '{"hydraulic"}', '{}')
ON CONFLICT DO NOTHING;

-- ── New QA pairs ─────────────────────────────────────────────
INSERT INTO qa_pairs (lift_models, question, answer_summary, drive_types, door_types)
VALUES ('{"Supermec"}', 'The Supermec activation board shows ERROR 4 with the red LED lit, CP and RE relays continuously supplied, and battery voltage nominal. What is the primary cause and how do I diagnose it?', 'ERROR 4 is a tracking error on the SED2 drive card: the encoder speed feedback deviates from the theoretical speed by more than the SGL_ERR (Parameter 74) threshold. The CP/RE relay status and battery voltage are unrelated — they reflect activation board battery health, not drive card state. Primary cause is encoder malfunction. Diagnose by: (1) checking encoder cable at J3 (shunt box) and J8 (activation board) for loose connections or damaged shielding; (2) verifying encoder supply voltage; (3) if cable is sound, the encoder has likely failed and needs replacement. Also check SGL_ERR setting — if too tight it causes nuisance trips.', '{"hydraulic"}', '{}')
ON CONFLICT DO NOTHING;
INSERT INTO qa_pairs (lift_models, question, answer_summary, drive_types, door_types)
VALUES ('{"Supermec"}', 'What component failure causes ERROR 4 on the Supermec SED2 drive card?', 'The primary component failure causing ERROR 4 is encoder malfunction. The SED2 drive card compares actual motor speed (from the encoder via J8 on the activation board) against the theoretical speed. When the difference exceeds the SGL_ERR threshold, ERROR 4 triggers and the red LED lights. Check the encoder cable (shielded, J3→J8), encoder supply voltage, and the encoder itself.', '{"hydraulic"}', '{}')
ON CONFLICT DO NOTHING;
INSERT INTO qa_pairs (lift_models, question, answer_summary, drive_types, door_types)
VALUES ('{"Supermec"}', 'Supermec ERROR 4 vs ERROR 1: what is the difference?', 'ERROR 1 on Supermec indicates low battery voltage (below SGL_B1x10 threshold) — yellow LED. ERROR 4 indicates tracking error too high — encoder actual speed differs from theoretical speed by more than SGL_ERR — red LED. They are independent faults; ERROR 1 relates to batteries, ERROR 4 to the encoder/drive card.', '{"hydraulic"}', '{}')
ON CONFLICT DO NOTHING;

COMMIT;
