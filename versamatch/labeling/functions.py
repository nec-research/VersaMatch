"""
       VersaMatch
	  
  File:     functions.py 
  Authors:  jonathan.fuerst@neclab.eu 
            mauricio.fadel@neclab.eu
            bin.cheng@neclab.eu

NEC Laboratories Europe GmbH, Copyright (c) 2023, All rights reserved.  

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
 
       PROPRIETARY INFORMATION ---  

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor. 

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.  

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.  

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.  

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""

from pandas.core.common import flatten
from nltk.stem import PorterStemmer
from fuzzywuzzy import fuzz
import itertools
import re
import numpy as np
from collections import Counter

@soft
def KF_class_name_equal(r):
    """
    Matches classes if they have the exact same and unique name.
    """
    if r.primary_texts_x["class_name"].lower() == r.primary_texts_y["class_name"].lower():
        return 1
    return -1

@soft
def KF_label_name_split_equal(r):
    """
    Matches classes based on splitting (e.g., for camel case names) and
    comparing their splits using lexical and semantic similarities, while
    considering the frequencies of the single splits in the corpus of the
    overall ontology.
    """
    threshold = 0.40

    keys = ["class_name_split", "label_split"]
    combinations = []
    for key_x in keys:
        if key_x in r.primary_texts_x:
            for key_y in keys:
                if key_y in r.primary_texts_y:
                    combinations.append((key_x, key_y))
    for c in combinations:
        key_x = c[0]
        key_y = c[1]
        name_x = [e for e in r.primary_texts_x[key_x].split() if len(e) > 2]
        name_y = [e for e in r.primary_texts_y[key_y].split() if len(e) > 2]

        common_ls = list(set(name_x) & set(name_y))
        for cw in common_ls:
            if name_x.index(cw) == name_y.index(cw):
                threshold = threshold - 0.1
        if r.primary_texts_variances_x[key_x.replace("_split", "_root")] not in common_ls or r.primary_texts_variances_y[key_y.replace("_split", "_root")] not in common_ls:
            threshold = threshold + 0.05
        if len(common_ls) == 0:
            continue
        x_unique = " ".join([w for w in name_x if w not in name_y])
        y_unique = " ".join([w for w in name_y if w not in name_x])
        x_unique_spacy = nlp(x_unique)
        y_unique_spacy = nlp(y_unique)
        if not x_unique_spacy.vector_norm or not y_unique_spacy.vector_norm:
            s = 0
        else:
            s = x_unique_spacy.similarity(y_unique_spacy)
        if s > 0.5:
            threshold = threshold - 0.1
        if len(common_ls) / len(set(name_x).union(name_y)) < threshold:
            continue
        comb_sorted = tuple(sorted(common_ls))
        if r.frequencies_x[key_x][comb_sorted] > 10:
            continue
        elif r.frequencies_y[key_y][comb_sorted] > 10:
            continue
        x_syn = []
        for vx in x_unique.split():
            for syn in wordnet.synsets(vx):
                for lm in syn.lemmas():
                    x_syn.append(lm.name())
        y_syn = []
        for vy in y_unique.split():
            for syn in wordnet.synsets(vy):
                for lm in syn.lemmas():
                    y_syn.append(lm.name())

        common_syn = list(set(x_syn) & set(y_syn))
        if len(common_syn) > 1:
            return 1
        if r.frequencies_x[key_x][comb_sorted] > 1:
            continue
        elif r.frequencies_y[key_y][comb_sorted] > 1:
            continue
        return 1
    return -1

@soft
def KF_class_name_split_spacy_distance(r):
    """
    Match classes based on semantic similarity, using split class names as they
    are more likely to be valid words that are contained in the training
    corpus.
    """
    # check if valid word, words that are not in corpus are not
    if r.primary_texts_spacy_x["class_name_split"].vector_norm and r.primary_texts_spacy_y["class_name_split"].vector_norm:
        s = r.primary_texts_spacy_x["class_name_split"].similarity(r.primary_texts_spacy_y["class_name_split"])
        if s > 0.95:
            return 1
        if s < 0.20:
            return 0
    return -1

@soft
def KF_class_name_split_stemmed_equal(r):
    """
    Splits and stemmes class names and then matches based on overlap of
    stemmmed splits.
    """
    porter = PorterStemmer()
    threshold = 0.4
    name_x = r.primary_texts_variances_x["class_name_split_stemmed"].split()
    name_y = r.primary_texts_variances_y["class_name_split_stemmed"].split()

    # single word exception
    if len(name_x) == 1 and len(name_y) == 1:
        if len(name_x[0]) > 3 and name_x[0] == name_y[0]:
            if r.frequencies_x["class_name_split_stemmed"][tuple([name_x[0]])] == 1 and r.frequencies_y["class_name_split_stemmed"][tuple([name_y[0]])] == 1:
                return 1
    common_ls = list(set(name_x) & set(name_y))
    if porter.stem(r.primary_texts_variances_x["class_name_root"]) not in common_ls or porter.stem(r.primary_texts_variances_y["class_name_root"]) not in common_ls:
        threshold = threshold + 0.1
    if len(common_ls) == 0 or len(common_ls) / len(set(name_x).union(name_y)) < threshold:
        return -1
    comb_sorted = tuple(sorted(common_ls))
    if r.frequencies_x["class_name_split_stemmed"][comb_sorted] > 1:
        return -1
    elif r.frequencies_y["class_name_split_stemmed"][comb_sorted] > 1:
        return -1
    return 1

@soft
def KF_primary_texts_equal_0(r):
    """
    Matches based on exact matches of primary texts from alignment profile
    (label, equivalent classes...)
    """
    keys_x = set(r.primary_texts_x.keys())
    keys_y = set(r.primary_texts_y.keys())

    common_keys = list(keys_x & keys_y)
    common_keys = [k for k in common_keys if "hasRelatedSynonym_name" not in k and "hasRelatedSynonym_name_split" not in k]

    for key in common_keys:
        if r.primary_texts_x[key].lower() == r.primary_texts_y[key].lower() and len(r.primary_texts_x[key]) > 0:
            return 1
    return -1

@soft
def KF_primary_texts_levenshtein_distance(r):
    """
    Compare strings using fuzzy string matching technique and vote based on
    threshold on distance.
    """
    upper_threshold = 85
    lower_threshold = 20
    primary_x = list(flatten(list([e.lower() for k, e in r.primary_texts_x.items() if "hasRelatedSynonym_name" not in k and "hasRelatedSynonym_name_split" not in k])))
    primary_y = list(flatten(list([e.lower() for k, e in r.primary_texts_y.items() if "hasRelatedSynonym_name" not in k and "hasRelatedSynonym_name_split" not in k])))
    combinations = list(itertools.product(primary_x, primary_y))
    ratios = []
    for v in combinations:
        x_clean = v[0]
        y_clean = v[1]
        digits_x = "".join(re.findall("\d+", x_clean))
        digits_y = "".join(re.findall("\d+", y_clean))
        if len(digits_x) >= 1 and len(digits_y) >= 1:
            if digits_x != digits_y:
                continue
        if len(x_clean) < 3 or len(y_clean) < 3:
            continue
        ratio = fuzz.ratio(x_clean, y_clean)
        if ratio > (upper_threshold + 10):
            return 1
        ratios.append(ratio)
    if not ratios:
        return 0
    overall_ratio = sum(ratios) / len(ratios)
    if overall_ratio > upper_threshold:
        return 1
    if overall_ratio < lower_threshold:
        return 0
    return -1

@soft
def KF_primary_texts_use_distance(r):
    """
    Matches based on USE distance of primary texts from alignment profile.
    """
    upper_threshold = 0.97
    lower_threshold = 0.20

    x_keys = [e[0] for e in r.primary_texts_x.items() if len(e[1]) > 2 and "hasRelatedSynonym_name" not in e[0]]
    y_keys = [e[0] for e in r.primary_texts_y.items() if len(e[1]) > 2 and "hasRelatedSynonym_name" not in e[0]]
    if "label" in x_keys and "label" in y_keys:
        x_keys = [e for e in x_keys if e not in ("class_name", "class_name_split")]
        y_keys = [e for e in y_keys if e not in ("class_name", "class_name_split")]
    combinations = list(itertools.product(x_keys, y_keys))
    i = len(x_keys) * len(y_keys)
    non_matches = 0
    for v in combinations:
        x_use = r.primary_texts_use_x[v[0]]
        y_use = r.primary_texts_use_y[v[1]]

        mag1 = np.linalg.norm(x_use[0])
        mag2 = np.linalg.norm(y_use[0])
        if (not mag1) or (not mag2):
            s = 0
        else:
            s = np.dot(x_use[0], y_use[0]) / (mag1 * mag2)

        if s > upper_threshold:
            return 1
        if s < lower_threshold:
            non_matches += 1
    if non_matches >= (i / 2):
        return 0
    return -1

@soft
def KF_primary_texts_synonyms(r):
    """
    Matches based on overlap in synonyms contained in alignment profile.
    """
    threshold = 1
    synonyms_x = set(flatten(list(r.primary_texts_synonyms_x.values())))
    synonyms_y = set(flatten(list(r.primary_texts_synonyms_y.values())))
    common = list(synonyms_x & synonyms_y)
    if len(synonyms_x) > 20 or len(synonyms_y) > 20:
        threshold = threshold + 1
    if len(common) >= threshold:
        return 1
    return -1

@soft
def KF_acronyms_split(r):
    """
    Matches acronyms based on regex, such as VLDB and Very Large Data Bases.
    """
    x = r.primary_texts_variances_x.get("class_name_acronym")
    y = r.primary_texts_variances_y.get("class_name_acronym")
    if x.lower() == r.primary_texts_y.get("class_name").lower() or y.lower() == r.primary_texts_x.get("class_name").lower():
        return 1
    common = Counter(x) & Counter(y)
    common_length = sum(common.values())
    if common_length < 3 or (common_length + 1 < len(x) or common_length + 1 < len(y)):
        return -1
    if r.frequencies_x["class_name_acronym"][x] > 2:
        return -1
    if r.frequencies_y["class_name_acronym"][y] > 2:
        return -1
    if (common_length >= len(x) and (len(y) - common_length) <= 1) or (common_length >= len(y) and (len(x) - common_length) <= 1):
        name_x = r.primary_texts_x["class_name_split"].split()
        name_y = r.primary_texts_y["class_name_split"].split()
        common_ls = list(set(name_x) & set(name_y))
        if len(common_ls) == 0:
            return -1
        comb_sorted = tuple(sorted(common_ls))
        if r.frequencies_x["class_name_split"][comb_sorted] > 2 and r.frequencies_y["class_name_split"][comb_sorted] > 2:
            return -1
        return 1
    return -1

@soft
def KF_superclasses_equal(r):
    """
     Matches based on superclasses equality.
    """
    threshold = 65
    if len(r.superclasses_x) > 0 and len(r.superclasses_y) > 0:
        x_sc = list(r.superclasses_x[0].keys())[0]
        y_sc = list(r.superclasses_y[0].keys())[0]
        if x_sc.lower() == y_sc.lower():
            primary_x = set(flatten(list(r.primary_texts_x.values())))
            primary_y = set(flatten(list(r.primary_texts_y.values())))
            for x in primary_x:
                for y in primary_y:
                    ratio = fuzz.ratio(x, y)
                    if ratio > threshold:
                        return 1
    return -1

@hard
def KF_subclasses_equal(r):
    """
    Matches based on subclasses equality.
    """
    threshold = 85
    if len(r.subclasses_x) > 0 and len(r.subclasses_y) > 0:
        x_sc = list(r.subclasses_x[0].keys())[0]
        y_sc = list(r.subclasses_y[0].keys())[0]
        if x_sc.lower() == y_sc.lower():
            primary_x = set(flatten(list(r.primary_texts_x.values())))
            primary_y = set(flatten(list(r.primary_texts_y.values())))
            for x in primary_x:
                for y in primary_y:
                    ratio = fuzz.ratio(x, y)
                    if ratio > threshold:
                        return 1
    return -1

@soft
def KF_wikidata(r):
    """
    Matches based on WikiData KB info.
    """
    if "label_split" not in r.primary_texts_x or "label_split" not in r.primary_texts_y:
        return -1
    if "label_split" not in r.primary_texts_x or "label_split" not in r.primary_texts_y:
        return -1
    if "wikidata_label" in r.background_knowledge_x and "wikidata_label" in r.background_knowledge_y:
        if r.background_knowledge_x["wikidata_label"] == r.background_knowledge_y["wikidata_label"]:
            return 1
    if "wikidata_aliases" in r.background_knowledge_y:
        if r.primary_texts_x["label_split"].lower() in [v.lower() for v in r.background_knowledge_y["wikidata_aliases"]]:
            return 1
    if "wikidata_aliases" in r.background_knowledge_x:
        if r.primary_texts_y["label_split"].lower() in [v.lower() for v in r.background_knowledge_x["wikidata_aliases"]]:
            return 1
    return -1

@hard
def KF_DomainKB(r):
    """
    Matches based on synonyms available in domain specific knowledge bases/ontologies.
    """
    if r.primary_texts_x["label_split"].lower() in [v.lower() for v in r.background_knowledge_y["DomainKB"]]:
        return 1
    if r.primary_texts_y["label_split"].lower() in [v.lower() for v in r.background_knowledge_x["DomainKB"]]:
        return 1
    return -1

@soft
def KF_root_nouns(r):
    """
    Matches based on equality of word roots.
    """

    root_x = r.primary_texts_variances_x.get("class_name_root")
    root_y = r.primary_texts_variances_y.get("class_name_root")

    if root_x.lower() == root_y.lower() and fuzz.ratio(r.name_x, r.name_y) > 90:
        return 1
    else:
        return -1
