#!/usr/bin/env python3

# check for the latest release at:  https://github.com/amikael/coding

# conllenc2.py v.2.0 (c) 2019 Anssi Yli-Jyrä
#
#    "supertag encoder and decoder of dependency graphs" 
#    - converts between the HEAD column and SuperTags in MISC column and NCRF++ like formats 
#      but with the new rope decomposition -based bracketing
#    - based on depconv.py  v.0.1 (c) 2015-2019 Anssi Yli-Jyrä 
#    - based on conllenc.py v.1.0 (c) 2019      Anssi Yli-Jyrä
#    - supporting also graph property annotation, codestring extraction and statistics
#
# Typical input file:
#
# 1	From	from	ADP	IN	_	3	case	_	_
# 2	the	the	DET	DT	-	3	det	_	_
# 3	AP	AP	PROPN	NNP	-	4	obl	_	_
# 4	comes	come	VERB	VBZ	-	0	root	_	_
# 5	this	this	DET	DT	-	6	det	_	_
# 6	story	story	NOUN	NN	-	4	nsubj	_	_
# 7	:	:	PUNCT	:	_	4	punct	_	_
# 
# Limitations:
# - the CoNLL input does not handle secondary dependencies (fix intended)
# - the CoNLL input does not handle semantic graph banks (fix in parallel versions)
# - the one-line encoding does not contain all information (lacking motivation)
#
# Features
# - Elliptic tokens (like 1.1, 1.2, etc.) are not supposed to be connected to other tokens
#   (this follows the specifications of U-CoNLL)

import pyconll
import re
import sys
import argparse
import string
import math

parser = argparse.ArgumentParser()
parser.add_argument('filename', help='Specified input files',nargs='*')
parser.add_argument('--input', help='Specified input file', dest="input")
parser.add_argument('--instring', action='count', help='Read tag strings instead of CoNLL')
parser.add_argument('--output', help='Print to a given output file', dest="output")
#
parser.add_argument('--max-cond', action='store', help='Stop after printing n sentences', type=int, dest="max")
parser.add_argument('--thick-cond', action='store', help='Sets minimum rope thickness', type=int, default=0, dest="thick")
parser.add_argument('--all-cond', action='count', help='No statistics on special classes of graphs', dest="all")
parser.add_argument('--nonx-cond', action='count', help='Restrict input to noncrossing graphs', dest="nonx")
parser.add_argument('--proj-cond', action='count', help='Restrict input to weakly projective, rooted graphs', dest="proj")
#
parser.add_argument('--voc-action', action='count', help='Print tag vocabulary', dest="voc")
parser.add_argument('--prop-action', action='count', help='Print computed properties for each graph', dest="prop")
parser.add_argument('--stats-action', action='count', help='Print statistics at the end', dest="stat")
parser.add_argument('--string-action', action='count', help='Produce the one-line encoding format', dest="string")
parser.add_argument('--misc-action', action='count', help='Produce encoding in MISC field', dest="misc")
parser.add_argument('--head-action', action='count', help='Restore HEADs from encoding', dest="head")
parser.add_argument('--nosupertags-action', action='count', help='Removes supertags from MISC field', dest="nosupertags")
parser.add_argument('--noheads-action', action='count', help='Removes heads from HEAD field', dest="noheads")
parser.add_argument('--nocodestring-action', action='count', help='Removes metafield', dest="nocodestring")
#
parser.add_argument('--indices-modif', action='count', help='Add edge indices to printed brackets', dest="indices")
parser.add_argument('--deprel-modif', action='count', help='Encode also the DEPREL', dest="deprel")
parser.add_argument('--pos-modif', action='store', help='Encode also the UPOS/XPOS', dest="uposxpos")
parser.add_argument('--wrap-modif', action='count', help='Split encoded string into lines', dest="wrap")
parser.add_argument('--oldproj-modif', action='count', help='--proj with zeroless bracketing', dest="oldproj")
parser.add_argument('--shifted-modif', action='count', help='word boundary shifted between closing and opening brackets', dest="shifted")
#
parser.add_argument('--tests', action='count', help='Run implicit unit tests')
parser.add_argument('--version', action='count', help='Print version number')
args = parser.parse_args()
if args.oldproj:
    args.proj = True
if args.input:
    args.filename += [args.input]
if args.filename == []:
    args.filename = ["-"]
if args.version:
    print("conllenc.py v.1.1 (c) 2019 Anssi Yli-Jyrä", file=sys.stderr)

def tint(st):
    assert('.' not in st)
    return int(st)

def edges(sentence):
    # compute a sorted list of edges of the underlying graph
    edges, edgelabel = [], {}
    for token in sentence:
        if "-" in token.id:
            continue

        # this could be rewritten more elegantly:
        token_head = token.head
        if token_head == None:
            continue
        if tint(token_head) == 0:
            continue
        elif tint(token_head) < tint(token.id):
            edge = (tint(token_head), tint(token.id))
        else:
            edge = (tint(token.id), tint(token_head))
        edges += [edge] 

        # add arc direction to edge labels
        if args.deprel and tint(token_head) > tint(token.id):
            edgelabel[edge] = token.deprel
        elif args.deprel:
            edgelabel[edge] = token.deprel
        else:
            edgelabel[edge] = ""

    edges.sort()
    return (edges, edgelabel)

def arcs(sentence):
    # compute a sorted list of arcs of the underlying graph
    arcs = []
    arclabel = {}
    for token in sentence:
        if "-" in token.id:
            continue
        token_head = token.head
        if token_head == None:
            token_head = str(int(float(token.id))*10)
        if tint(token_head) != 0:
            arc   = (tint(token_head), tint(token.id))
            arcs += [arc]
    arcs.sort()
    return arcs

# ROPE DECOMPOSITION

def proper_cover_edges(edges):
    # compute the set of such edges (i,j) such that i is minimal for j and j is maximal for i
    maximal_j_for_i = {}
    minimal_i_for_j = {}
    for (i,j) in edges:
        if i not in maximal_j_for_i or j > maximal_j_for_i[i]:
            maximal_j_for_i[i] = j
        if j not in minimal_i_for_j or i < minimal_i_for_j[j]:
            minimal_i_for_j[j] = i
    pce = []
    for (i,j) in edges:
        if minimal_i_for_j[j] == i and maximal_j_for_i[i] == j:
            pce += [(i,j)]
    pce.sort()
    return pce

def not_covered(edges, pce):
    # compute the set of edges that are not yet covered by pce
    j_for_i = {}
    i_for_j = {}
    for (i,j) in pce:
        j_for_i[i] = j
        i_for_j[j] = i
    residual = []
    for (i,j) in edges:
        if i in j_for_i and j <= j_for_i[i]:
            continue
        if j in i_for_j and i >= i_for_j[j]:
            continue
        residual += [(i,j)]
    return residual

def proper_rope_cover(edges):
    # compute the set of edges that form the proper rope cover of the graph
    # this is based on a theorem in my FSMNLP submission
    prc   = []
    pce   = []
    edges = sorted(edges)
    while edges:
        edges = not_covered(edges, pce)
        # print("R=", pce, " E=",edges)
        pce   = proper_cover_edges(edges)
        prc   = prc + pce
    prc.sort()
    return prc

def transpose(A):
    return [(j,i) for (i,j) in A]

def digraph(sentence):
    Vn = [tint(token.id) for token in sentence if "-" not in token.id]
    A  = arcs(sentence)
    return (Vn,A)

def underlying_graph(sentence):
    Vn = [tint(token.id) for token in sentence if "-" not in token.id]
    (E,Elabel) = edges(sentence)
    return (Vn,E,Elabel)

def prepend(elem,list):
    list[0:0] = [elem]

# TESTS
 
def is_nonx(arcs):
    if arcs == []:
        return (True," (no arcs)")
    rmost = {}
    for (i,j) in arcs:
        (i,j) = (min(i,j),max(i,j))
        if i not in rmost:
            rmost[i] = j
        else:
            rmost[i] = max(rmost[i],j)
    for (i,j) in arcs:
        (i,j) = (min(i,j),max(i,j))
        for k in range(i+1,j):
            if k in rmost and j < rmost[k]:
                return (False," (crossing edges ({},{}),({},{}))".format(i,j,k,rmost[k]))
    return (True,"")

def is_nonx_wproj_with_root(sentence,arcs,nonx):
    # this is conjunction of three properties:
    #  - noncrossing (because we wanted to have nested statistics)
    #  - head is not covered by a daughter's dependency 
    #    ('weakly projective', see Yli-Jyrä and Gómez-Rodríguez 2017
    #  - root is not covered by any dependency
    # if the graph is a tree, this coincides with 'projective tree'
    #
    # note that projetivity does not imply treeness (see Yli-Jyrä 2005)!
    if arcs == []:
        return (True,"")
    if not nonx:
        return (False," (crossing)")
    head, root = {}, 0
    for tok in sentence:
        head[tint(tok.id)] = tint(tok.head)
        if tint(tok.head) == 0:
            if root:
                return (False," (no unique root)")                
            root = tint(tok.id)
    for (h,d) in arcs:
        if h < head[h] and head[h] < d:
            return (False," (not weakly projective ({},{},{}))".format(h,head[h],d))
        if d < head[h] and head[h] < h:
            return (False," (not weakly projective ({},{},{}))".format(d,head[h],h))
        (i,j) = (min(h,d),max(h,d))
        if i < root and root < j:
            return (False," (root covered ({},{},{}))".format(i,root,j))
    return (True,"")


def is_bad(sentence):
    for token in sentence:
        if "." in token.id:
            return True
    return False
            
def has_heads(sentence):
    for tok in sentence:
        if tok.head != '0':
            return True
    return False

def has_supertags(sentence):
    for tok in sentence:
        for tag in tok.misc:
            if (len(tag) >= len('SuperTag') and tag[0:len('SuperTag')] == 'SuperTag'):
                return True
    return False

# TRANSITION-BASED ENCODER

class RopeDecomp:
    def __init__(self, Vn, R, AL, AR, IL, IR):
        self.Vn = Vn
        self.R  = R
        self.AL = AL
        self.AR = AR
        self.IL = IL
        self.IR = IR
    def undo(self):
        Rmap = {i:j for (i,j) in self.R}
        ILA  = [(i, Rmap[j]) for (i,j) in self.IL]
        IRA  = [(i, Rmap[j]) for (i,j) in self.IR]
        AA   = sorted(transpose(self.AL) + self.AR + transpose(ILA) + IRA)
        return (self.Vn,self.AA)

lb = "⦗" # LEFT  BLACK TORTOISE SHELL BRACKET (U+2997, Ps): ⦗
rb = "⦘" # RIGHT BLACK TORTOISE SHELL BRACKET (U+2998, Pe): ⦘
trantable = str.maketrans('','',' ')
fstr = "{}{}({},{})" if args.indices else "{}{}"

def encode_ropedecomp_as_codestring(ropedecomp,Elabel,sentence):
    Vn,R,AL,AR,IL,IR = ropedecomp.Vn, ropedecomp.R, ropedecomp.AL, ropedecomp.AR, ropedecomp.IL, ropedecomp.IR
    S1, S2, st = [], [], ""
    ALR, ILR   = AL + AR, IL + IR
    Ri, Rmap   = [i for (i,j) in R], {i:j for (i,j) in R}
    thickness = depth2 = 0
    nonx = True
    # THIS MAY BREAK WHEN "-"
    for [token,j] in [[tok,tint(tok.id)] for tok in sentence if "-" not in tok.id]:  # NEXT
        i = -1
        if not args.shifted and token != sentence[0]:
            st += "·"
        thickness = max(thickness, len(S1))
        ii = [i for (i,jp) in ALR if jp == j] + [h for (jp,h) in ILR if jp == j]
        if ii:
            min_i = min(ii)
            while i != min_i:
                (i,left) = S1.pop()
                if (i,j) in R: # REDUCE                    
                    #print("reduce",S1,"   ",S2)
                    if (i,j) in AR:
                        st += fstr.format("⟧>",Elabel[(i,j)],i,j) 
                    else:
                        st += fstr.format("<⟧",Elabel[(i,j)],i,j) 
                    nonx = nonx and not len(S2)  # if stack2 nonempty => crossing
                else:
                    # This code assumes no edge is bidirectional
                    if (i,j) in AL:
                        st += fstr.format("<"+rb,Elabel[(i,j)],i,j)
                        nonx = nonx and left # crossing [ and ]-arcs  ⟦ [ ] 
                    elif (i,j) in AR:
                        st += fstr.format(rb+">",Elabel[(i,j)],i,j)
                        nonx = nonx and left # crossing [ and ]-arcs  ⟦ [ ] 
                    elif not args.oldproj:
                        st += rb+"₀"
                    prepend((i,left),S2) # PASS
                    #print("pass",S1,"   ",S2)
        depth2 = max(depth2, len(S2))
        nonx = nonx and depth2 <= 1
        if args.shifted and token != sentence[-1]:
            st += "·"
        while S2: # INSERT            
            (i,left) = S2.pop(0)
            # This code assumes no edge is bidirectional
            if (j,i) in IL:
                st += fstr.format("<"+lb,Elabel[(j,Rmap[i])],j,Rmap[i]) 
                S1.append((i,False)) # expecting right-side brackets ⟦...[[[...
            elif (j,i) in IR:
                st += fstr.format(lb+">",Elabel[(j,Rmap[i])],j,Rmap[i]) 
                S1.append((i,False)) # expecting right-side brackets ⟦...[[[...
            elif not args.oldproj:
                st += lb+"₀"
                S1.append((i,left))
            else:
                st += ""
                S1.append((i,left))
            #print("insert",S1,"   ",S2)
        if j in Ri: # SHIFT
            if (j,Rmap[j]) in AR:
                st += fstr.format("⟦>",Elabel[(j,Rmap[j])],j,Rmap[j]) 
            else:
                st += fstr.format("<⟦",Elabel[(j,Rmap[j])],j,Rmap[j]) 
            S1.append((j,True)) # expecting left-side brackets ⟦]]]...
            #print("shift",S1,"   ",S2)
    tags, spc = "", ""
    for (tok,tag) in zip(sentence,st.split("·")):
        pos = tok.upos if args.uposxpos == "UPOS" else tok.xpos
        pos = "_" if pos == None else pos
        assert(tok.form != None)   # frm = tok.form if tok.form != None else "(None)" 
        tags += spc + tok.form + "\t" + pos + "\t" + tag + "_" + tok.deprel
        spc  = "·"
    return (tags,thickness,depth2,nonx)

def generic_encode(sentence):
    (graph_Vn,E,Elabel) = underlying_graph(sentence)
    R = proper_rope_cover(E)
    (digraph_Vn, A) = digraph(sentence)
    assert(graph_Vn == digraph_Vn)
    ropedecomp = map_digraph_and_prc_to_ropedecomp(digraph_Vn,A,R)
    (codestr,thickness,depth2,nonx) = encode_ropedecomp_as_codestring(ropedecomp,Elabel,sentence)
    return (codestr,thickness,depth2,nonx,A)

def parse_fulltag(fulltag):
    # format1  HEAD_DEPREL - not implemented
    # format2  HEADOFFSET_DEPREL - not implemented
    # format3  POS_POSOFFSET_DEPREL - not implemented
    # format4  SUPERTAG_DEPREL - not implemented
    # format5: SUPERTAG_DEPREL:
    subtags = fulltag.split("_")
    assert(len(subtags) == 2)
    (supertag,deprel) = subtags
    return (deprel,supertag)

def parse_node(node):
    # _token fulltag 
    # fulltag _token
    elems = [e for e in node.split("\t") if e != ""]
    assert(len(elems) == 3)
    token, pos, fulltag = elems
    if args.uposxpos == "XPOS":
        upos, xpos = "_", pos
    else:
        upos, xpos = pos, "_"
    (deprel,supertag) = parse_fulltag(fulltag)
    return (token,fulltag,upos,xpos,deprel,supertag)
    
def codestr_to_conll(codestr):
    id, positions, conll = 1, codestr.split("·"), ""
    for fulltag in positions:
        (token,fulltag,upos,xpos,deprel,supertag) = parse_node(fulltag)
        conll += "{}\t{}\t{}\t{}\t{}\t_\t0\t{}\t_\t_\n".format(id,token,token,upos,xpos,deprel)
        id   += 1
    return conll

def supertags_to_codestr(sentence):
    codestr = ""
    spc = ""
    for tok in sentence:
        for tag in tok.misc:
            # This is a pyconll-dependent hack
            if len(tag) > len('SuperTag') and tag[0:len('SuperTag')] == 'SuperTag':
                code = tag[len('SuperTag'):]
                codestr += spc + code
                spc = "·"
    return codestr

def map_digraph_and_prc_to_ropedecomp(Vn, A, R): # max_vertex, proper rope cover, all arcs
    # compute IL and IR
    RTmap = {j:i for (i,j) in R}
    AT    = transpose(A)
    IL    = [(i,RTmap[j]) for (i,j) in AT if i<j and j in RTmap and RTmap[j] < i]
    IR    = [(i,RTmap[j]) for (i,j) in A  if i<j and j in RTmap and RTmap[j] < i]
    if args.tests:
        IL.sort()
        IR.sort()
        for (i,h) in IL+IR:
            assert(h < i)
    # compute AL and AR
    Ri   = [i   for (i,j) in R]
    Rmap = {i:j for (i,j) in R}
    ILA  = [(i, Rmap[j]) for (i,j) in IL]
    AL   = [(i,j) for (i,j) in AT if i<j and i in Ri and (i,j) not in ILA]
    IRA  = [(i, Rmap[j]) for (i,j) in IR]
    AR   = [(i,j) for (i,j) in A  if i<j and i in Ri and (i,j) not in IRA]
    AL.sort()
    AR.sort()
    RD   = RopeDecomp(Vn,R,AL,AR,IL,IR)
    if args.tests:
        (Vnx,Ax) = RD.undo()
        assert(Vn == Vnx)
        assert(A == Ax)
    return RopeDecomp(Vn,R,AL,AR,IL,IR)

def decode_codestr_to_ropedecomp(codestr):

    def pass_(S1,S2):
        i = S1.pop()
        prepend(i,S2)
        return i
    def insert_(S1,S2):
        i = S2.pop(0)
        S1.append(i)
        return i

    codestr = re.findall("⦗₀|\[\⁰|<⦗|\[<|⦗>|\[|⦘₀|\]\⁰|⦘>|\]>|<⦘|\]|⟧>|<⟧|⟧|⟦>|<⟦|⟦|·",codestr)
    offset = 1 if args.shifted else 0
    Vn, S1, S2, AL, AR, IL, IR, A, R = [1], [], [], [], [], [], [], [], []
    j, root, nopass = 1, 0, False
    for a in codestr:
        # print(a)
        if "·" in a:
            while S2 != []:
                insert_(S1,S2) #print("added ⦗₀")
            j, Vn, nopass = j+1, Vn + [j+1], True
        elif "⟦" in a:
            while S2 != []:
                insert_(S1,S2) #print("added ⦗₀")
            S1.append(j-offset)
        elif "⟧" in a: 
            if S1 != []: # popping from nonempty list
                i = S1.pop()
                R += [(i,j)]
                AR,AL = (AR+[(i,j)],AL) if '>' in a else (AR,AL+[(i,j)]) if '<' in a else (AR,AL)
        elif "]" in a or "⦘" in a:
            if S1 != []: # popping from nonempty list
                i = pass_(S1,S2)
                nopass = False
                AR,AL = (AR+[(i,j)],AL) if ">" in a else (AR,AL+[(i,j)]) if "<" in a else (AR,AL)
        elif "[" in a or "⦗" in a:
            if S1 == [] and S2 == []: # popping from empty stacks
                continue
            if nopass and S2 == [] and S1 != []: # recover from empty S2
                pass_(S1,S2) #print("added ⦘₀")
                nopass = False
            i = insert_(S1,S2)
            IR,IL = (IR+[(j-offset,i)],IL) if ">" in a else (IR,IL+[(j-offset,i)]) if "<" in a else (IR,IL)
    return RopeDecomp(Vn, sorted(R), sorted(AL), sorted(AR), sorted(IL), sorted(IR))

def print_statline(max_thickness,thicknesses,all,ofile):
    if max_thickness == 0:
        return
    print("# thickness   1 ", end="", file=ofile)
    for i in range(2,max_thickness+1):
        print("{:9d}".format(i), end=" ", file=ofile)
    print("\n# N   ", end="", file=ofile)
    for i in range(1,max_thickness+1):
        print("{:9d}".format(thicknesses[i]), end=" ", file=ofile)
    print("\n# cumN", end="", file=ofile)
    sum = 0 
    for i in range(1,max_thickness+1):
        sum += thicknesses[i]
        print("{:9d}".format(sum), end=" ", file=ofile)
    print("\n# %   ", end="", file=ofile)
    sum = 0 
    for i in range(1,max_thickness+1):
        sum += thicknesses[i]
        perc = (100.0 * sum) / all
        print("{:8.2f}%".format(perc), end=" ", file=ofile)
    print("", file=ofile)
            
def print_sentence(sentence,ofile):
    if args.string:
        if args.prop:
            print("#", sentence.meta_value("properties"), "\n", file=ofile)
        if args.wrap:
            print("\n".join(sentence.meta_value("codestring").split("·")), file=ofile)
            print("", file=ofile)
        else:
            print(" ".join(sentence.meta_value("codestring").split("\t")), file=ofile)
    elif args.head or args.misc or not args.stat and not args.voc:
        print(sentence.conll(), file=ofile)
        print("", file=ofile)

class Sent:
    def __init__(self, sentence):
        self.sentence = sentence
        self.printable = True
    def ropedecomp_to_heads(self, ropedecomp):
        # - noncrossing       holds by tag vocabulary
        # - weak projectivity holds by tag vocabulary
        # - one-headedness    holds by tag vocabulary for unshifted and projective
        #                     holds by heads filling priority 
        # - acyclicity        holds by cycle cutting
        # - one-rootedness    holds by forest connection
        def reach(i,reached):
            if i not in reached:
                reached.add(i)
                if i in deps:
                    for j in deps[i]:
                        reach(j,reached)
        def leftcorner(i):
            lc = i
            if i in deps:
                for j in deps[i]:
                    lc = min(lc,leftcorner(j))
            return lc
        def makedeps(head):
            deps = {}
            for i in head:
                j = head[i]
                if j in deps:
                    deps[j] += [i]
                else:
                    deps[j] = [i]
            for i in deps:
                deps[i].sort()
            return deps

        Rmap = {i:j for (i,j) in ropedecomp.R}
        AIR = [(i,Rmap[j]) for (i,j) in ropedecomp.IR] + ropedecomp.AR
        AIL = [(i,Rmap[j]) for (i,j) in ropedecomp.IL] + ropedecomp.AL

        # turn arcs to head links: => enforce one_headedness, ignoring the additional edge links
        head = {}
        for (i,j) in AIR:
            if j not in head or head[j] > i:  # postprocess: choose the first head
                head[j] = i
        for (i,j) in AIL:
            if i not in head or head[i] > j:  # postprocess: choose the first head
                head[i] = j
        deps = makedeps(head)
        
        reached, root = set([]), 0
        for i in range(1,tint(self.sentence[-1].id)+1):
            if i not in head:      # reach trees if any
                reach(i,reached)
                root = root if root else i
        for i in range(1,tint(self.sentence[-1].id)+1):
            if i not in reached:
                root = root if root else i
                head.pop(i)        # break remaining cycles
                reach(i,reached)
        deps = makedeps(head)
        for i in range(1,tint(self.sentence[-1].id)+1):
            if i not in head and i != root:
                h = 1 if i == 1 else leftcorner(i)-1
                # print(i," has been assigned a new head: ",h)
                head[i] = h # connect forest
        # store to CoNLL data model: 
        for tok in self.sentence:
            i = tint(tok.id)
            if i in head:
                tok.head = "{}".format(head[i])
    def sentence_rm_heads(self):
        for token in self.sentence:
            token.head = "0"
    def sentence_rm_supertags(self):
        for tok in self.sentence:
            misc =  {}
            for tag in tok.misc:
                if not (len(tag) >= len('SuperTag') and tag[0:len('SuperTag')] == 'SuperTag'):
                    misc[tag] = tok.misc[tag]
            tok.misc = misc
    def set_meta_codestring(self, codestr):
        self.sentence.set_meta("codestring",codestr) # "".join(codestr))
    def process_properties(self,stats,thickness,depth2,nonx,A):
        (verinonx,nonxfailure) = is_nonx(A)
        assert(verinonx == nonx)
        (veriproj,projfailure) = is_nonx_wproj_with_root(self.sentence,A,verinonx)
        stats.update(thickness,depth2,verinonx,veriproj)
        self.printable = ((not args.nonx or verinonx) and 
                          (not args.proj or veriproj) and
                          (args.thick == 0 or args.thick < thickness))
        if args.prop:
            value = ""
            if verinonx:
                value = "noncrossing, "
            else:
                value = "crossing{}, ".format(nonxfailure)
            if veriproj:
                value += "projective, "
            else:
                value += "not projective{}, ".format(projfailure)
            value += "rope thickness {}, ".format(thickness)
            value += "auxiliary stack size {}"  .format(depth2)
            self.sentence.set_meta("properties",value)
    def process_tag_vocabulary(self,stats,codestr):
        nodes = codestr.split(" · ")
        for node in nodes:
            (token,token,fulltag,upos,xpos,deprel,supertag) = parse_node(node)
            if fulltag not in stats.voc:
                stats.voc[fulltag] = 1
            else:
                stats.voc[fulltag] += 1
            if supertag not in stats.minivoc:
                stats.minivoc[supertag] = 1
            else:
                stats.minivoc[supertag] += 1                    
    def codestr_to_misc(self, codestr):
        codestr = re.findall("((?:⦗₀|\[\⁰|<⦗|\[<|⦗>|\[|⦘₀|\]\⁰|⦘>|\]>|<⦘|\]|⟧>|<⟧|⟧|⟦>|<⟦|⟦|·)(?:\([0-9]+,[0-9]+\))?)",codestr)
        supertags = "".join(codestr).split("·")
        for supertag, token in zip(supertags, self.sentence):
            token.misc['SuperTag'+supertag] = None
    def process_sent(self, stats, ofile):
        if args.head or args.instring:
            codestring = (self.sentence.meta_value("codestring") 
                          if args.instring else
                          supertags_to_codestr(self.sentence))
            self.ropedecomp_to_heads(decode_codestr_to_ropedecomp(codestring))
        if stats.going_to_produce_supertags or stats.going_to_process_properties:
            (codestr,thickness,depth2,nonx,A) = generic_encode(self.sentence)
            self.set_meta_codestring(codestr)
            if args.misc:
                pass
            self.codestr_to_misc(codestr)
            if stats.going_to_process_properties:
                self.process_properties(stats,thickness,depth2,nonx,A)
            if self.printable and args.voc:
                self.process_tag_vocabulary(stats,codestr)
        if self.printable:
            if args.nosupertags:
                self.sentence_rm_supertags()
            if args.misc and args.instring and not args.head or args.noheads:
                self.sentence_rm_heads()
            if args.nocodestring and self.sentence.meta_present('codestring'):
                self.sentence._meta.pop('codestring')
            print_sentence(self.sentence, ofile)
            return 1
        return 0
      
class Stats:
    def __init__(self):
        self.sents = self.bads = self.all = self.nonxes = self.projs = self.printed = 0
        self.max_thickness = self.max_depth2 = 0
        self.thicknesses, self.thicknesses_nx, self.thicknesses_pj = [0]*21, [0]*21, [0]*21
        self.going_to_produce_supertags  = args.string or args.misc
        self.going_to_process_properties = (args.prop or args.stat or args.nonx or args.proj or 
                                            args.thick > 0 or args.tests or args.voc)
        self.voc, self.minivoc = {}, {}
    def update(self,thickness,depth2,verinonx,veriproj):
        self.max_thickness = max(thickness, self.max_thickness)
        self.max_depth2 = max(depth2, self.max_depth2)
        self.thicknesses[thickness] += 1
        if verinonx:
            self.thicknesses_nx[thickness] += 1
            self.nonxes += 1
        if verinonx and veriproj:
            self.thicknesses_pj[thickness] += 1
            self.projs += 1
    def print(self,ofile):
        print("# STATISTICS:", file=ofile)
        if self.sents-self.all > 0:
            print("# all graphs  "," {:5.1f}%  {:7d} (with ellipsis {:d})".
                  format(100.0 * self.sents / self.sents, self.sents, self.sents-self.all), file=ofile)
            print("#   (the current script cannot process ellipsis)")
        print("# processed   "," {:5.1f}%  {:7d} (with crossing {:d})".
              format(100.0 * self.all / self.sents, self.all, self.all-self.nonxes), file=ofile)
        if not args.nonx and not args.proj:
            print_statline(self.max_thickness,self.thicknesses, self.all, ofile)
        if not args.proj and not args.all:
            print("#\n# noncrossing "," {:5.1f}%  {:7d} (with not weak projective {:d})".
                  format(100.0 * self.nonxes / self.sents, self.nonxes, self.nonxes - self.projs), file=ofile)
            print_statline(self.max_thickness,self.thicknesses_nx, self.all, ofile)
        if not args.nonx and not args.all:
            print("#\n# nx & w.proj "," {:5.1f}%  {:7d}".
                  format(100.0 * self.projs / self.sents, self.projs), file=ofile)
            print_statline(self.max_thickness, self.thicknesses_pj, self.all, ofile)
        print("# ", file=ofile)
    def process_corpus(self, corpus, ofile):
        for sentence in corpus:
            self.sents += 1
            if is_bad(sentence):
                self.bads += 1
                continue
            self.all += 1
            sent = Sent(sentence)
            self.printed += sent.process_sent(self, ofile)
            if args.max and args.max <= self.printed:
                break

def read_encoded_corpus_from_file(file):
    corpus = pyconll.unit.Conll("")
    contents = open(file,"r")
    codestr = ""
    for codestrpart in contents:
        if args.wrap:
            if codestrpart != "\n":
                if codestr:
                    codestr += "·"
                codestr += codestrpart[:-1]
                continue
        elif codestr[-1] == '\n':
            codestr = codestr[:-1]
        sentence = pyconll.unit.Sentence(codestr_to_conll(codestr))
        sentence.set_meta("codestring"," ".join(codestr.split("\t")))
        corpus.insert(corpus.__len__(),sentence)
        codestr = ""
    return corpus

def main():
    stats = Stats()
    ofile = open(args.output,"w") if args.output else open("/dev/stdout","w")
    for f in args.filename:
        if f == '-':
            f = '/dev/stdin'
        if args.instring:
            corpus = read_encoded_corpus_from_file(f)
        else:
            corpus = pyconll.load_from_file(f)
        stats.process_corpus(corpus, ofile)
    if args.stat:
        stats.print(ofile)
    if args.voc:
        print("# VOCABULARY: (size {})".format(len(stats.voc)), file=ofile)
        print("# ", stats.voc, file=ofile)
        cross_sum = 0
        for i in stats.voc:
            cross_sum += stats.voc[i]
        entropy = 0
        for i in stats.voc:
            entropy -= stats.voc[i] * math.log(stats.voc[i] / cross_sum) / math.log(2)
        print("# UNIGRAM ENTROPY OF TAGGING: ",entropy," ({} bits per tag, total {} tags)".format(entropy/cross_sum,cross_sum))
        if len(stats.minivoc):
            print("# MINIVOCABULARY: (size {})".format(len(stats.minivoc)), file=ofile)
            print("# ", stats.minivoc, file=ofile)

main()
