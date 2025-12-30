import requests
import json

def get_tags(text: str) -> str:
    """get tags for the old french phrase using Udpipe API."""
    API_URL = "https://lindat.mff.cuni.cz/services/udpipe/api/process"
    data = {
        "data": text,
        "model": "old_french-profiterole-ud-2.17-251125",
        "tokenizer":"",
        "tagger":"",
        "parser":""
    }
    response = requests.post(API_URL, data=data)
    response.raise_for_status()
    output = response.json()["result"]
    return output

def convert_tags_to_dict(udpipe_output: str) -> dict:
    """Convert Udpipe output to a dictionary of words and their POS tags."""
    tag_dict = {}
    lines = udpipe_output.strip().split("\n")
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 4:
            word = parts[1]
            tag_dict[word] = {
                "pos_tag": parts[3],
                "x_pos": parts[4],
                "head": parts[6],
                "dep_rel": parts[7]}
            # if parts[3] in ['VERB', 'NOUN', 'ADJ']:
            #     tag_dict[word] = {
            #     "pos_tag": parts[3],
            #     "x_pos": parts[4],
            #     "head": parts[6],
            #     "dep_rel": parts[7]
            # }

    return tag_dict

def lookup_word(entries, query, pos, visited = set(), max_depth=3):
    results = {query: {}}
    

    def _lookup(q, depth):
        if q in visited or depth > max_depth:
            return
        visited.add(q)

        for entry in entries:
            if entry.get("pos") != pos:
                continue

            forms = {f.get("form") for f in entry.get("forms", [])}
            if q != entry.get("word") and q not in forms:
                continue
            
            word = entry.get("word")
            visited.add(word)
            bucket = results[query].setdefault(word, {
                "entry": entry,
                "pos": pos,
                "glosses": [],
                "alt_words": []
            })

            for sense in entry.get("senses", []):
                bucket["glosses"].extend(sense.get("glosses", []))

                for alt in sense.get("alt_of", []):
                    alt_word = alt.get("word")
                    if alt_word:
                        bucket["alt_words"].append(alt_word)
                        _lookup(alt_word, depth + 1)

    _lookup(query, 1)
    return results, visited




if __name__ == "__main__":
    
    original="Et quant Lanselos l’a abatu, il nel regarde plus, non plus que s’il ne l’eüst onques veü, mais la u il voit les autres cevaliers, ki ja estoient monté et estoient issu des paveillons tout apareillié de ferir et de grever Lanselot se il peüssent. Lanselos, ki de nule riens ne les doute, ains les bee tous a metre a desconfiture, s’il onques puet, lors laisse courre tout maintenant."
    # original = "Et de tant li fu il bien avenu k’il n’avoit encore mie son glaive brisié, ains en fiert le premier k’il encontre si durement k’il li fait vuidier les arcons. Et s’il avoit fait mal a l’autre cevalier, encore fist il pis a cestui de cest caup."
    words = "cevalier, chevalier, corrocié, courechié"
    tags = get_tags(original)
    dict_tags = convert_tags_to_dict(tags)

    with open("kaikki.org-dictionary-OldFrench-words.jsonl", "r") as f:
        entries = [json.loads(line) for line in f]
    visited_words = set()
    for word, tag_info in dict_tags.items():
        if tag_info['pos_tag'] in ['VERB', 'NOUN', 'ADJ']:
            results, visited_words = lookup_word(entries, word, tag_info['pos_tag'].lower(), visited_words, max_depth=2)
            for query, words in results.items():
                if not words:
                    print(f"{query}, no result")
                    continue

                print(f"{query}:")
                for w, data in words.items():
                    glosses = "; ".join(set(data["glosses"]))
                    print(f"    {w}, {data['pos']}, {glosses}")
                print()

            
            
                