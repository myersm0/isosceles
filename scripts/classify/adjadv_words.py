"""Per-word ADJ↔ADV guidance for the adjadv classifier."""

swing_words = {
	"tout": {
		"adj": 'determiner/adjective modifying a noun: "toute la ville", "tous les jours", "toute seule"',
		"adv": 'intensifier before ADJ/ADV (invariable or agreeing): "tout entière", "tout doucement", "toute honteuse" (fem before consonant)',
	},
	"même": {
		"adj": 'adjective meaning "same" or emphatic: "le même jour", "elle-même"',
		"adv": 'adverb meaning "even": "même lui", "même pas", "pas même"',
	},
	"petit": {
		"adj": 'adjective modifying noun: "un petit jardin", "la petite fille"',
		"adv": 'rare, in fixed expressions: "petit à petit"',
	},
	"seul": {
		"adj": 'adjective/predicate: "un seul homme", "elle est seule", "seul dans la nuit"',
		"adv": 'rare adverbial use: "seul compte le résultat" (only the result matters)',
	},
	"avant": {
		"adj": 'adjective in compounds or position: "roue avant", "partie avant"',
		"adv": 'adverb of time/place: "la nuit d\'avant", "en avant", "bien avant"',
	},
	"haut": {
		"adj": 'modifying noun: "une haute montagne", "à voix haute", "la marée haute"',
		"adv": 'modifying verb: "parler haut", "monter haut", "haut placé"',
	},
	"fort": {
		"adj": 'modifying noun or predicate: "un fort accent", "elle est forte", "une forte somme"',
		"adv": 'intensifier or manner: "fort ému", "fort bien", "crier fort"',
	},
	"long": {
		"adj": 'modifying noun: "un long chemin", "une longue attente"',
		"adv": 'in fixed expressions: "en dire long", "en savoir long", "tout le long"',
	},
	"possible": {
		"adj": 'modifying noun or predicate: "une solution possible", "c\'est possible"',
		"adv": 'rare, in superlative constructions: "le plus vite possible"',
	},
	"vrai": {
		"adj": 'modifying noun or predicate: "un vrai ami", "c\'est vrai"',
		"adv": 'rare, sentence adverb: "vrai, il avait raison" (= truly)',
	},
	"plein": {
		"adj": 'modifying noun or predicate: "un verre plein", "la lune pleine"',
		"adv": 'preposition-like: "plein de gens", "plein les poches", "en plein jour"',
	},
	"juste": {
		"adj": 'modifying noun or predicate: "une voix juste", "c\'est juste"',
		"adv": 'meaning "only/exactly": "juste un instant", "juste à temps", "il vient juste de partir"',
	},
	"cher": {
		"adj": 'modifying noun or predicate: "un ami cher", "la vie est chère"',
		"adv": 'modifying verb: "coûter cher", "payer cher", "vendre cher"',
	},
	"droit": {
		"adj": 'modifying noun: "le côté droit", "la ligne droite", "il est droit"',
		"adv": 'modifying verb (direction): "aller droit", "marcher droit", "tout droit"',
	},
	"super": {
		"adj": 'predicate or modifier: "c\'est super", "un super film"',
		"adv": 'intensifier: "super bien", "super content"',
	},
	"chaud": {
		"adj": 'modifying noun or predicate: "un repas chaud", "l\'eau est chaude"',
		"adv": 'modifying verb: "avoir chaud", "manger chaud", "il fait chaud"',
	},
	"dur": {
		"adj": 'modifying noun or predicate: "un travail dur", "la terre est dure"',
		"adv": 'modifying verb: "travailler dur", "frapper dur", "cogner dur"',
	},
	"net": {
		"adj": 'modifying noun or predicate: "un refus net", "la coupure est nette"',
		"adv": 'modifying verb: "refuser net", "s\'arrêter net", "couper net"',
	},
	"large": {
		"adj": 'modifying noun or predicate: "une large rue", "le fleuve est large"',
		"adv": 'modifying verb: "voir large", "calculer large"',
	},
	"demi": {
		"adj": 'after noun: "une heure et demie", "midi et demi"',
		"adv": 'before adjective (hyphenated): "demi-mort", "à demi ouvert"',
	},
	"arrière": {
		"adj": 'positional: "la roue arrière", "le siège arrière"',
		"adv": 'directional: "en arrière", "revenir en arrière"',
	},
	"pire": {
		"adj": 'comparative: "le pire ennemi", "c\'est pire"',
		"adv": 'rare adverbial: "tant pire" (archaic for tant pis)',
	},
	"bis": {
		"adj": 'modifier: "pain bis", "12 bis"',
		"adv": 'repetition marker: "bis !" (encore)',
	},
	# Manual additions — common 19th-c. swing words not in GSD as both
	"soudain": {
		"adj": 'modifying noun: "un vide soudain", "une peur soudaine", "un bruit soudain"',
		"adv": 'sentence adverb meaning "suddenly": "soudain, il se leva", "soudain elle parut"',
	},
	"bas": {
		"adj": 'modifying noun or predicate: "une voix basse", "le plafond est bas"',
		"adv": 'modifying verb: "parler bas", "mettre bas", "tomber plus bas"',
	},
	"clair": {
		"adj": 'modifying noun or predicate: "une voix claire", "l\'eau est claire"',
		"adv": 'modifying verb: "voir clair", "parler clair", "il fait clair"',
	},
	"court": {
		"adj": 'modifying noun or predicate: "un chemin court", "la robe est courte"',
		"adv": 'modifying verb: "couper court", "tourner court", "rester court"',
	},
	"faux": {
		"adj": 'modifying noun or predicate: "une fausse note", "c\'est faux"',
		"adv": 'modifying verb: "chanter faux", "sonner faux", "jouer faux"',
	},
}
