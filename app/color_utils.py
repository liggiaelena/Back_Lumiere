def hex_to_rgb(hex_str: str) -> tuple:
    h = hex_str.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def color_delta(hex1: str, hex2: str) -> float:
    r1, g1, b1 = hex_to_rgb(hex1)
    r2, g2, b2 = hex_to_rgb(hex2)
    return ((0.299*(r1-r2))**2 + (0.587*(g1-g2))**2 + (0.114*(b1-b2))**2)**0.5

def build_final_report(region_results: dict) -> dict:
    regioes = region_results

    tons_fitz = [r["tom_fitzpatrick"] for r in regioes.values()]
    tom_geral = max(set(tons_fitz), key=tons_fitz.count)

    subtoms = [r["subtom"] for r in regioes.values()]
    subtom_geral = max(set(subtoms), key=subtoms.count)

    pares = [
        ("testa",  "bochecha_e"),
        ("testa",  "bochecha_d"),
        ("nariz",  "bochecha_e"),
        ("queixo", "testa"),
    ]

    comparacoes = {}
    for r1, r2 in pares:
        if r1 in regioes and r2 in regioes:
            delta = color_delta(regioes[r1]["tom_hex"], regioes[r2]["tom_hex"])
            comparacoes[f"{r1}_vs_{r2}"] = {
                "delta": round(delta, 2),
                "nivel": "alto" if delta > 20 else "moderado" if delta > 10 else "baixo"
            }

    todas_imperf = []
    for region, data in regioes.items():
        for imp in data.get("imperfeicoes", []):
            todas_imperf.append({**imp, "regiao": region})

    return {
        "tom_geral_fitzpatrick": tom_geral,
        "subtom_predominante":   subtom_geral,
        "regioes":               regioes,
        "comparacao_tons":       comparacoes,
        "imperfeicoes":          todas_imperf,
    }
