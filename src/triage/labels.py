"""Mapeamento das categorias do dataset para níveis de urgência."""

URGENCY_MAP = {
    # urgente: risco à vida ou piora rápida
    "Heart hurts": "urgente",
    "Hard to breath": "urgente",
    "Internal pain": "urgente",
    "Open wound": "urgente",
    "Infected wound": "urgente",
    # atencao: merece consulta logo, mas não é emergência
    "Feeling dizzy": "atencao",
    "Blurry vision": "atencao",
    "Body feels weak": "atencao",
    "Head ache": "atencao",
    "Stomach ache": "atencao",
    "Cough": "atencao",
    "Feeling cold": "atencao",
    "Ear ache": "atencao",
    "Injury from sports": "atencao",
    "Emotional pain": "atencao",
    # normal: queixa leve ou crônica, consulta de rotina
    "Back pain": "normal",
    "Neck pain": "normal",
    "Shoulder pain": "normal",
    "Knee pain": "normal",
    "Joint pain": "normal",
    "Muscle pain": "normal",
    "Foot ache": "normal",
    "Skin issue": "normal",
    "Acne": "normal",
    "Hair falling out": "normal",
}
