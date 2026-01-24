#!/usr/bin/env python3
"""
Build parallel index mapping French Maupassant stories to McMaster English translations.

Usage:
    python parallel_index.py check
    python parallel_index.py build
"""

import re
import sys
import unicodedata
from pathlib import Path

base_dir = Path(__file__).parent.parent
fr_txt_dir = base_dir / "data" / "maupassant_fr" / "txt"
en_txt_dir = base_dir / "data" / "maupassant_en" / "txt"

title_mapping = """
Le donneur d'eau bénite (Contes divers 1875-1880),The dispenser of holy water
Le mariage du lieutenant Laré (Contes divers 1875-1880),The lieutenant Lare's marriage
Boule de suif (Contes divers 1875-1880),Boule de suif
Les dimanches d'un bourgeois de Paris (Contes divers 1875-1880),Sundays of a bourgeois
Jadis (Contes divers 1875-1880),The love of long ago
La maison Tellier (La maison Tellier),The maison Tellier
Les tombales (La maison Tellier),Tombstones
Sur l'eau (La maison Tellier),On the river
Histoire d'une fille de ferme (La maison Tellier),The story of a farm girl
En famille (La maison Tellier),A family affair
Le papa de Simon (La maison Tellier),Simon's papa
Une partie de campagne (La maison Tellier),A country excursion
Au printemps (La maison Tellier),In the spring
La femme de Paul (La maison Tellier),Paul's mistress
Mademoiselle Fifi (Mademoiselle Fifi),Mademoiselle Fifi
Madame Baptiste (Mademoiselle Fifi),Madame Baptiste
La rouille (Mademoiselle Fifi),Rust
Marroca (Mademoiselle Fifi),Marroca
La bûche (Mademoiselle Fifi),The log
La relique (Mademoiselle Fifi),The relic
Le lit (Mademoiselle Fifi),The bed
Réveil (Mademoiselle Fifi),The awakening
Une ruse (Mademoiselle Fifi),An artifice
À cheval (Mademoiselle Fifi),That costly ride
Mots d'amour (Mademoiselle Fifi),Words of Love
Une aventure parisienne (Mademoiselle Fifi),An adventure in Paris
Deux amis (Mademoiselle Fifi),Two Friends
Nuit de Noël (Mademoiselle Fifi),Christmas Eve
Le remplaçant (Mademoiselle Fifi),The substitute
Le gâteau (Contes divers 1882),The cake
L'aveugle (Contes divers 1882),The blind man
Magnétisme (Contes divers 1882),Magnetism
En voyage (Contes divers 1882),The mountain pool
Un bandit corse (Contes divers 1882),The corsican bandit
La veillée (Contes divers 1882),A dead woman's secret
Rêves (Contes divers 1882),Dreams
Confession d'une femme (Contes divers 1882),A wife's confession
Clair de lune (Contes divers 1882),Moonlight
Une passion (Contes divers 1882),A passion
Correspondance (Contes divers 1882),The impolite sex
Le baiser (Contes divers 1882),The kiss
Ma femme (Contes divers 1882),My wife
Rouerie (Contes divers 1882),Woman's wiles
Yveline Samoris (Contes divers 1882),Yvette Samoris
La bécasse (Contes de la bécasse),The snipe
Ce cochon de Morin (Contes de la bécasse),That pig of a Morin
La folle (Contes de la bécasse),The mad woman
Pierrot (Contes de la bécasse),Pierrot
Menuet (Contes de la bécasse),Minuet
Farce normande (Contes de la bécasse),A normandy joke
Les sabots (Contes de la bécasse),The wooden shoes
La rempailleuse (Contes de la bécasse),Lasting Love
En mer (Contes de la bécasse),At sea
Un normand (Contes de la bécasse),Father Matthew
Le testament (Contes de la bécasse),The will
Aux champs (Contes de la bécasse),The adopted son
Un coq chanta (Contes de la bécasse),A cock crowed
Un fils (Contes de la bécasse),The son
Saint-Antoine (Contes de la bécasse),Saint Anthony
L'aventure de Walter Schnaffs (Contes de la bécasse),Walter Schnaff's Adventure
Clair de lune (Clair de lune),Clair de lune
Un coup d'État (Clair de lune),An affair of State
Le loup (Clair de lune),The wolf
L'enfant (Clair de lune),A wedding gift
La reine Hortense (Clair de lune),Queen Hortense
Le pardon (Clair de lune),Forgiveness
La légende du Mont Saint-Michel (Clair de lune),Legend of Mont St. Michel
Une veuve (Clair de lune),A widow
Mademoiselle Cocotte (Clair de lune),Mademoiselle Cocotte
Les bijoux (Clair de lune),The false gems
Apparition (Clair de lune),The apparition
La porte (Clair de lune),The door
Moiron (Clair de lune),Moiron
Nos lettres (Clair de lune),Our letters
La nuit (Clair de lune),The night
Auprès d'un mort (Contes divers 1883),Beside Schopenhauer's corpse
Le père Judas (Contes divers 1883),Old Judas
Le père Milon (Contes divers 1883),Father Milon
L'ami Joseph (Contes divers 1883),Friend Joseph
L'orphelin (Contes divers 1883),The orphan
Un duel (Contes divers 1883),A duel
Une soirée (Contes divers 1883),A queer night in Paris
Humble drame (Contes divers 1883),A humble drama
Le vengeur (Contes divers 1883),His avenger
L'attente (Contes divers 1883),Mother and son
Première neige (Contes divers 1883),The first snowfall
La farce (Contes divers 1883),A uncomfortable bed
Miss Harriet (Miss Harriet),Miss Harriet
Denis (Miss Harriet),Denis
L'âne (Miss Harriet),The donkey
La ficelle (Miss Harriet),The piece of string
Garçon, un Bock !... (Miss Harriet),Waiter, a "Bock"
Regret (Miss Harriet),Regret
Mon oncle Jules (Miss Harriet),My uncle Jules
La mère Sauvage (Miss Harriet),Mother Sauvage
Les sœurs Rondoli (Les sœurs Rondoli),The Rondoli sisters
La patronne (Les sœurs Rondoli),My landlady
Le petit fût (Les sœurs Rondoli),The little cask
Lui ? (Les sœurs Rondoli),The terror
Mon oncle Sosthène (Les sœurs Rondoli),My uncle Sosthenes
Le mal d'André (Les sœurs Rondoli),What was really the matter with Andrew
Le pain maudit (Les sœurs Rondoli),The accursed bread
Un sage (Les sœurs Rondoli),A philosopher
Le parapluie (Les sœurs Rondoli),The umbrella
Le verrou (Les sœurs Rondoli),Always lock the door!
Rencontre (Les sœurs Rondoli),A meeting
Suicides (Les sœurs Rondoli),Suicides
Décoré ! (Les sœurs Rondoli),The legion of honor
Châli (Les sœurs Rondoli),Châli
Yvette (Yvette),Yvette
L'abandonné (Yvette),Abandoned
Les idées du Colonel (Yvette),The Colonel's ideas
Promenade (Yvette),A stroll
Le garde (Yvette),The gamekeeper
Berthe (Yvette),Bertha
Lettre trouvée sur un noyé (Contes divers 1884),Found on a drowned man
Misti (Contes divers 1884),Misti
L'horrible (Contes divers 1884),The horrible
Le tic (Contes divers 1884),The spasm
La tombe (Contes divers 1884),The grave
Le bûcher (Contes divers 1884),A cremation
Rose (Contes du jour et de la nuit),Humiliation
Le père (Contes du jour et de la nuit),The father
L'aveu (Contes du jour et de la nuit),Confessing
La parure (Contes du jour et de la nuit),The Necklace
Le bonheur (Contes du jour et de la nuit),Happiness
Le vieux (Contes du jour et de la nuit),The moribund
Un lâche (Contes du jour et de la nuit),A coward
L'ivrogne (Contes du jour et de la nuit),The drunkard
Une vendetta (Contes du jour et de la nuit),The vendetta
Coco (Contes du jour et de la nuit),Coco
La main (Contes du jour et de la nuit),The hand
Le gueux (Contes du jour et de la nuit),The beggar
Un parricide (Contes du jour et de la nuit),A parricide
Le petit (Contes du jour et de la nuit),The child
La roche aux guillemots (Contes du jour et de la nuit),The penguins' rock
Tombouctou (Contes du jour et de la nuit),Timbuctoo
Adieu (Contes du jour et de la nuit),Farewell
Souvenir (Contes du jour et de la nuit),A recollection
La confession (Contes du jour et de la nuit),A sister's confession
Fini (Contes divers 1885),All over
Mes vingt-cinq jours (Contes divers 1885),My twenty-five days
Monsieur Parent (Monsieur Parent),Monsieur Parent
La bête à Maît' Belhomme (Monsieur Parent),Belhomme's beast
L'inconnue (Monsieur Parent),The unknown
Le baptême (Monsieur Parent),The christening
Imprudence (Monsieur Parent),Indiscretion
Un fou (Monsieur Parent),The diary of a madman
L'épingle (Monsieur Parent),Fascination
Découverte (Monsieur Parent),Discovery
Petit soldat (Monsieur Parent),Two little soldiers
Toine (Toine),Toine
L'ami Patience (Toine),Friend Patience
L'homme-fille (Toine),The effeminates
La moustache (Toine),The mustache
La dot (Toine),The dowry
Le protecteur (Toine),The patron
La chevelure (Toine),A tress of hair
Le père Mongilet (Toine),Old Mongilet
L'armoire (Toine),The wardrobe
Les prisonniers (Toine),The prisoners
La confession (Toine),A father's confession
La mère aux monstres (Toine),A mother of monsters
La confession de Théodule Sabot (Toine),Theodule Sabot's confession
La petite Roque (La petite Roque),Little Louise Roque
L'épave (La petite Roque),The wreck
L'ermite (La petite Roque),The hermit
Mademoiselle Perle (La petite Roque),Mademoiselle Pearl
Rosalie Prudent (La petite Roque),Rosalie Prudent
Sauvée (La petite Roque),Saved
Madame Parisse (La petite Roque),Madame Parisse
Julie Romain (La petite Roque),Julie Romaine
Le père Amable (La petite Roque),Old Amable
La question du latin (Contes divers 1886),The question of latin
Le fermier (Contes divers 1886),The farmer's wife
Le Horla (Le Horla),The Horla
Amour (Le Horla),Love
Le trou (Le Horla),The fishing hole
Clochette (Le Horla),Clochette
Le marquis de Fumerol (Le Horla),The marquis de Fumerol
Le signe (Le Horla),The signal
Le diable (Le Horla),The devil
Les rois (Le Horla),Epiphany
Au bois (Le Horla),In the Wood
Une famille (Le Horla),A family
Joseph (Le Horla),Joseph
L'auberge (Le Horla),The inn
Le vagabond (Le Horla),A vagabond
Étrennes (Contes divers 1887),New Year's gift
Madame Hermet (Contes divers 1887),Madame Hermet
Le voyage du Horla (Contes divers 1887),The trip of the Horla
Le rosier de Madame Husson (Le rosier de Madame Husson),Madame Husson's rosier
Le modèle (Le rosier de Madame Husson),The model
La baronne (Le rosier de Madame Husson),The baroness
Une vente (Le rosier de Madame Husson),A sale
La Martine (Le rosier de Madame Husson),Martine
Une soirée (Le rosier de Madame Husson),The wrong house
Allouma (La main gauche),Allouma
Hautot père et fils (La main gauche),Hautot senior and Hautot junior
Boitelle (La main gauche),Boitelle
L'ordonnance (La main gauche),The orderly
Le lapin (La main gauche),The rabbit
Les épingles (La main gauche),The double pins
Duchoux (La main gauche),Duchoux
Le rendez-vous (La main gauche),The assignation
Le port (La main gauche),The port
La morte (La main gauche),Was it a dream?
Alexandre (Contes divers 1889),Alexandre
L'endormeuse (Contes divers 1889),The magic couch
Le colporteur (Contes divers 1889),The peddler
Après (Contes divers 1889),After
L'inutile beauté (L'inutile beauté),Useless Beauty
Le champ d'oliviers (L'inutile beauté),The olive grove
Mouche (L'inutile beauté),Fly
Le noyé (L'inutile beauté),The parrot
L'épreuve (L'inutile beauté),The test
Le masque (L'inutile beauté),The mask
Un portrait (L'inutile beauté),A portrait
L'infirme (L'inutile beauté),The cripple
Les vingt-cinq francs de la supérieure (L'inutile beauté),The twenty-five francs of the mother-superior
Un cas de divorce (L'inutile beauté),A divorce case
Qui sait ? (L'inutile beauté),Who knows?
""".strip()

fr_filename_aliases = {
	"Le mal d'André": "malandre",
	"Mes vingt-cinq jours": "25jours",
	"Au bord du lit": "bordlit",
	"Jour de fête": "jourfete",
	"Les vingt-cinq francs de la supérieure": "25francs",
}


def normalize_for_filename(title):
	name = title.lower()
	name = unicodedata.normalize("NFD", name)
	name = "".join(c for c in name if unicodedata.category(c) != "Mn")
	name = re.sub(r"[^a-z0-9]+", "_", name)
	name = re.sub(r"_+", "_", name).strip("_")
	return name


def word_count(filepath):
	if filepath is None:
		return 0
	try:
		text = filepath.read_text(encoding="utf-8")
		return len(text.split())
	except:
		return 0


def parse_mapping():
	entries = []
	for line in title_mapping.split("\n"):
		line = line.strip()
		if not line:
			continue
		match = re.match(r"^(.+?)\s*\(([^)]+)\)\s*,\s*(.+)$", line)
		if match:
			entries.append({
				"fr_title": match.group(1).strip(),
				"collection": match.group(2).strip(),
				"en_title": match.group(3).strip(),
			})
	return entries


def find_file(directory, title, aliases=None):
	if not directory.exists():
		return None

	if aliases and title in aliases:
		alias = aliases[title]
		path = directory / f"{alias}.txt"
		if path.exists():
			return alias

	normalized = normalize_for_filename(title)

	for f in directory.glob("*.txt"):
		if f.stem == normalized:
			return f.stem

	for f in directory.glob("*.txt"):
		if f.stem.startswith(normalized[:10]) or normalized.startswith(f.stem[:10]):
			return f.stem

	for f in directory.glob("*.txt"):
		f_norm = f.stem.replace("_", "")
		t_norm = normalized.replace("_", "")
		if f_norm in t_norm or t_norm in f_norm:
			return f.stem

	return None


def cmd_check():
	entries = parse_mapping()

	fr_files = {f.stem for f in fr_txt_dir.glob("*.txt")} if fr_txt_dir.exists() else set()
	en_files = {f.stem for f in en_txt_dir.glob("*.txt")} if en_txt_dir.exists() else set()

	print(f"French files: {len(fr_files)}")
	print(f"English files: {len(en_files)}")
	print(f"Mapping entries: {len(entries)}")
	print()

	matched = 0
	fr_missing = []
	en_missing = []

	for entry in entries:
		fr_file = find_file(fr_txt_dir, entry["fr_title"], fr_filename_aliases)
		en_file = find_file(en_txt_dir, entry["en_title"])

		if fr_file and en_file:
			matched += 1
		else:
			if not fr_file:
				fr_missing.append(entry["fr_title"])
			if not en_file:
				en_missing.append(entry["en_title"])

	print(f"Matched pairs: {matched}")
	print()

	if fr_missing:
		print(f"=== French titles not found ({len(fr_missing)}) ===")
		for t in fr_missing[:20]:
			print(f"  {t}")
		if len(fr_missing) > 20:
			print(f"  ... and {len(fr_missing) - 20} more")
		print()

	if en_missing:
		print(f"=== English titles not found ({len(en_missing)}) ===")
		for t in en_missing[:20]:
			print(f"  {t}")
		if len(en_missing) > 20:
			print(f"  ... and {len(en_missing) - 20} more")


def cmd_build():
	entries = parse_mapping()

	output = []
	for entry in entries:
		fr_file = find_file(fr_txt_dir, entry["fr_title"], fr_filename_aliases)
		en_file = find_file(en_txt_dir, entry["en_title"])

		fr_path = (fr_txt_dir / f"{fr_file}.txt") if fr_file else None
		en_path = (en_txt_dir / f"{en_file}.txt") if en_file else None

		output.append({
			"fr_file": fr_file or "",
			"en_file": en_file or "",
			"fr_title": entry["fr_title"],
			"en_title": entry["en_title"],
			"collection": entry["collection"],
			"fr_words": word_count(fr_path),
			"en_words": word_count(en_path),
		})

	out_path = base_dir / "data" / "parallel_index.tsv"
	with open(out_path, "w", encoding="utf-8") as f:
		f.write("fr_file\ten_file\tfr_title\ten_title\tcollection\tfr_words\ten_words\n")
		for row in output:
			f.write(f"{row['fr_file']}\t{row['en_file']}\t{row['fr_title']}\t{row['en_title']}\t{row['collection']}\t{row['fr_words']}\t{row['en_words']}\n")

	matched = [row for row in output if row["fr_file"] and row["en_file"]]
	fr_only = [row for row in output if row["fr_file"] and not row["en_file"]]
	en_only = [row for row in output if row["en_file"] and not row["fr_file"]]

	print(f"Wrote {out_path}")
	print(f"Total entries: {len(output)}")
	print(f"Matched pairs: {len(matched)}")
	print(f"French only: {len(fr_only)}")
	print(f"English only: {len(en_only)}")

	if matched:
		total_fr = sum(row["fr_words"] for row in matched)
		total_en = sum(row["en_words"] for row in matched)
		ratio = total_en / total_fr if total_fr > 0 else 0
		print(f"\nMatched pairs word counts:")
		print(f"  French: {total_fr:,} words")
		print(f"  English: {total_en:,} words")
		print(f"  Ratio (en/fr): {ratio:.3f}")


def main():
	if len(sys.argv) < 2:
		print(__doc__)
		return

	cmd = sys.argv[1]
	if cmd == "check":
		cmd_check()
	elif cmd == "build":
		cmd_build()
	else:
		print(f"Unknown command: {cmd}")
		print(__doc__)


if __name__ == "__main__":
	main()
