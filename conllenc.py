#!/usr/bin/env python3

# conllenc.py v.1 (c) 2019 Anssi Yli-Jyrä
#
#    "supertag encoder and decoder of dependency graphs" 
#    - currently restricted to (primary) dependencies in CoNLL-U files
#    - converts between the HEAD column and SuperTags in MISC column 
#    - based on depconv v.0.1 (c) 2015-2019 Anssi Yli-Jyrä (and Carlos Gómez-Rodríguez)
#    - supporting also graph property annotation, codestring extraction and statistics
#    
# Typical usage:
# 
#   conllenc.py          en-ud-dev.conllu           --conll > en-ud-dev.supertags.conllu
#   conllenc.py --decode en-ud-dev.supertags.conllu --conll > en-ud-dev.restored.conllu
#   conllenc.py --stat   en-ud-dev.conllu
#   conllenc.py          en-ud-dev.conllu --string  --conll | egrep '^$|# codestring = '
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
# Typical output with --conll:
#
# 1	From	from	ADP	IN	_	0	case	_	SuperTag⟦
# 2	the	the	DET	DT	_	0	det	_	SuperTag]⁰[<
# 3	AP	AP	PROPN	NNP	_	0	obl	_	SuperTag⟧⟦
# 4	comes	come	VERB	VBZ	_	0	root	_	SuperTag⟧⟦
# 5	this	this	DET	DT	_	0	det	_	SuperTag⟦
# 6	story	story	NOUN	NN	_	0	nsubj	_	SuperTag⟧]>[⁰
# 7	:	:	PUNCT	:	_	0	punct	_	SuperTag⟧>
#
# Typical output with --conll --indices --deprel:
#
# 1	From	from	ADP	IN	_	0	case	_	SuperTag⟦
# 2	the	the	DET	DT	_	0	det	_	SuperTag]⁰[<det(2,3)
# 3	AP	AP	PROPN	NNP	_	0	obl	_	SuperTag⟧case(1,3)⟦
# 4	comes	come	VERB	VBZ	_	0	root	_	SuperTag⟧obl(3,4)⟦
# 5	this	this	DET	DT	_	0	det	_	SuperTag⟦
# 6	story	story	NOUN	NN	_	0	nsubj	_	SuperTag⟧det(5,6)]>nsubj(4,6)[⁰
# 7	:	:	PUNCT	:	_	0	punct	_	SuperTag⟧>punct(4,7)
#
# Typical output with --conll --string and egrep post-filtering and with --deprel --pos --indices options:
#
#   codestring =  _From ⟦· ]⁰ _the [< · ⟧ _AP ⟦· ⟧ _comes ⟦· _this ⟦· ⟧ ]> _story [⁰ · ⟧> _:
#
#   codestring =  ADP.IN_From ⟦· ]⁰ DET.DT_the [<det(2,3) · ⟧case(1,3) PROPN.NNP_AP ⟦· ⟧obl(3,4) 
#   VERB.VBZ_comes ⟦· DET.DT_this ⟦· ⟧det(5,6) ]>nsubj(4,6) NOUN.NN_story [⁰ · ⟧>punct(4,7) PUNCT.:_:
#
# Combining options:
# 
#   enrich the input with codestring, properties, supertags and commandline options:
#     conllenc.py --conll --string --prop --copy --trace en_lines-ud-dev.conllu 
#   get subsets with --nonx --proj
#     conllenc.py --conll --string --prop --copy --trace en_lines-ud-dev.conllu --proj
# 
#   print all statistics:
#     conllenc.py --stat en_lines-ud-dev.conllu 
#   get less statistics with --all --nonx --proj
#
#   move head links to supertags:
#     conllenc.py --conll en_lines-ud-dev.conllu > supertags.conllu
#   get richer supertags with --pos --deprel --indices
#     conllenc.py --conll en_lines-ud-dev.conllu --deprel --ind
#
#   and back to head links:
#     cat supertags.conllu | conllenc.py - --decode --conll 
#     cat supertags.conllu | conllenc.py - --decode --conll | diff - en_lines-ud-dev.conllu
#    
#   run additional internal unit tests with --tests 
#
# Limitations:
# - the CoNLL input does not handle secondary dependencies (fix intended)
# - the CoNLL input does not handle ellipsis tokens (lacking precise specifications)
# - the CoNLL input does not handle semantic graph banks (fix in parallel versions)
# - the one-line encoding does not contain all information (lacking motivation)

import pyconll
import re
import sys
import argparse
import string

parser = argparse.ArgumentParser()
parser.add_argument('filename', help='The name of the file to transform',nargs='*')
parser.add_argument('--trace', action='count', help='Print command line switches in the beginning')
parser.add_argument('--decode', action='count', help='Decode heads from MISC field')
parser.add_argument('--copy', action='count', help='Keep both supertags and heads')
parser.add_argument('--indices', action='count', help='Add edge indices to printed brackets')
parser.add_argument('--deprel', action='count', help='Print also the arc label')
parser.add_argument('--pos', action='count', help='Print also the POS tag')
parser.add_argument('--conll', action='count', help='Produce the CoNLL format')
parser.add_argument('--string', action='count', help='Produce the one-line encoding format')
parser.add_argument('--prop', action='count', help='Print computed properties for each graph')
parser.add_argument('--stat', action='count', help='Print statistics at the end')
parser.add_argument('--all', action='count', help='Do not print stats anything but all graphs')
parser.add_argument('--nonx', action='count', help='Do not print stats anything but noncrossing graphs')
parser.add_argument('--proj', action='count', help='Do not print stats anything but projective graphs')
parser.add_argument('--tests', action='count', help='Run implicit unit tests')
parser.add_argument('--version', action='count', help='Print version number')
args = parser.parse_args()

if args.version:
    print("conllenc.py v.1 (c) 2019 Anssi Yli-Jyrä")

# CONLL TO GRAPH
def tint(st):
    # we assume that token ids have no more than 10 subtoken numbers
    if '.' in st:
        if len(st) < 3 or st[-2] != '.':
            print("BAD TOKEN INDEX ",st)
            exit(1)
#        return int(float(st)*10)
#    return int(st)*10
        return int(float(st))
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
#           edgelabel[edge] = "<"+token.deprel
            edgelabel[edge] = token.deprel
        elif args.deprel:
#           edgelabel[edge] = token.deprel+">"
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

def undo_rope_decomposition(Vn,R,AL,AR,IL,IR):
    Rmap = {i:j for (i,j) in R}
    ILA  = [(i, Rmap[j]) for (i,j) in IL]
    IRA  = [(i, Rmap[j]) for (i,j) in IR]
    AA   = sorted(transpose(AL) + AR + transpose(ILA) + IRA)
    return (Vn,AA)

def rope_decomposition(Vn, R, A):
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
    if args.tests:
        AL.sort()
        AR.sort()
        (Vnx,Ax) = undo_rope_decomposition(Vn,R,AL,AR,IL,IR)
        assert(Vn == Vnx)
        assert(A == Ax)
    return (Vn,R,AL,AR,IL,IR)

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

# this is conjunction of three properties:
#  - noncrossing (because we wanted to have nested statistics)
#  - head is not covered by a daughter's dependency 
#    ('weakly projective', see Yli-Jyrä and Gómez-Rodríguez 2017
#  - root is not covered by any dependency
# if the graph is a tree, this coincides with 'projective tree'

def is_nonx_wproj_with_root(sentence,arcs,nonx):
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

def rm_heads(sentence):
    for token in sentence:
        token.head = "0"

def rm_supertags(sentence):
    for tok in sentence:
        misc =  {}
        for tag in tok.misc:
            if not (len(tag) >= len('SuperTag') and tag[0:len('SuperTag')] == 'SuperTag'):
                misc[tag] = tok.misc[tag]
        tok.misc = misc

# TRANSITION-BASED ENCODER

def outtoken(token):
    form = "_" + token.form + " " if token.form != None else "_ " 
    # form = token.id
    return token.upos + "." + token.xpos + form if args.pos else form

def transitions(Vn,R,AL,AR,IL,IR,Elabel):
    trantable = str.maketrans('','',' ')
    S1, S2, st = [], [], ""
    ALR, ILR   = AL + AR, IL + IR
    Ri, Rmap   = [i for (i,j) in R], {i:j for (i,j) in R}
    fstr = "{}{}({},{}) " if args.indices else "{}{} "
    depth1 = depth2 = 0
    nonx = True
    # print("init",S1,"   ",S2)
    for [token,j] in [[tok,tint(tok.id)] for tok in sentence if "-" not in tok.id]:  # NEXT
        i   = -1
        st += "· "
        cat = ""
        depth1 = max(depth1, len(S1))
        ii   = [i for (i,jp) in ALR if jp == j] + [h for (jp,h) in ILR if jp == j]
        if ii:
            min_i = min(ii)
            while i != min_i:
                (i,left) = S1.pop()
                if (i,j) in R: # REDUCE                    
                    # print("reduce",S1,"   ",S2)
                    if (i,j) in AR:
                        st  += fstr.format("⟧>",Elabel[(i,j)],i,j) 
                        cat += fstr.format("⟧>",Elabel[(i,j)],i,j) 
                    else:
                        st  += fstr.format("⟧",Elabel[(i,j)],i,j) 
                        cat += fstr.format("⟧",Elabel[(i,j)],i,j) 
                    nonx = nonx and not len(S2)  # if stack2 nonempty => crossing
                else:
                    # This code assumes no edge is bidirectional
                    if (i,j) in AL:
                        st  += fstr.format("]",Elabel[(i,j)],i,j)
                        cat += fstr.format("]",Elabel[(i,j)],i,j)
                        nonx = nonx and left # crossing [ and ]-arcs  ⟦ [ ] 
                    elif (i,j) in AR:
                        st  += fstr.format("]>",Elabel[(i,j)],i,j)
                        cat += fstr.format("]>",Elabel[(i,j)],i,j)
                        nonx = nonx and left # crossing [ and ]-arcs  ⟦ [ ] 
                    else:
                        st  += "]⁰ "
                        cat += "]⁰ "
                    prepend((i,left),S2) # PASS
                    # print("pass",S1,"   ",S2)
        depth2 = max(depth2, len(S2))
        nonx = nonx and depth2 <= 1
        st  += outtoken(token)
        cat += ""
        while S2: # INSERT            
            (i,left) = S2.pop(0)
            # This code assumes no edge is bidirectional
            if (j,i) in IL:
                st  += fstr.format("[<",Elabel[(j,Rmap[i])],j,Rmap[i]) 
                cat += fstr.format("[<",Elabel[(j,Rmap[i])],j,Rmap[i]) 
                S1.append((i,False)) # expecting right-side brackets ⟦...[[[...
            elif (j,i) in IR:
                st  += fstr.format("[",Elabel[(j,Rmap[i])],j,Rmap[i]) 
                cat += fstr.format("[",Elabel[(j,Rmap[i])],j,Rmap[i]) 
                S1.append((i,False)) # expecting right-side brackets ⟦...[[[...
            else:
                st  += "[⁰ "
                cat += "[⁰ "
                S1.append((i,left))
            # print("insert",S1,"   ",S2)
        if j in Ri: # SHIFT
            st  += "⟦"
            cat += "⟦"
            S1.append((j,True)) # expecting left-side brackets ⟦]]]...
            # print("shift",S1,"   ",S2)
        cat = cat.translate(trantable)
        token.misc['SuperTag'+cat] = None
    return (st[1:],depth1,depth2,nonx)

def getcodestr(sentence):
    codestr = ""
    spc = ""
    for tok in sentence:
        for tag in tok.misc:
            # This is a pyconll-dependent hack
            if len(tag) > len('SuperTag') and tag[0:len('SuperTag')] == 'SuperTag':
                code = tag[len('SuperTag'):]
                codestr += spc + code
                spc = "·"
    return re.findall("\[\⁰|\[<|\[|\]\⁰|\]>|\]|⟦·|·|⟧>|⟧",codestr)

def decode(codestr):
    Vn, S1, S2, AL, AR, IL, IR, A, R = [1], [], [], [], [], [], [], [], []
    j, root = 1, 0
    for a in codestr:
        if a[-1] == "·":
            if a == "⟦·":
                S1.append(j)
            j, Vn = j+1, Vn + [j+1]
        elif a[0] == "⟧":
            i = S1.pop()
            R += [(i,j)]
            if a == "⟧>":
                AR += [(i,j)]
            else:
                AL += [(i,j)]
        elif a[0] == "]":
            i = S1.pop()
            prepend(i,S2)
            if a == "]>":
                AR += [(i,j)]
            elif a == "]":
                AL += [(i,j)]
        elif a[0] == "[":
            i = S2.pop(0)
            S1.append(i)
            if a == "[<":
                IL += [(j,i)]
            elif a == "[":
                IR += [(j,i)]
    return (Vn,sorted(R),sorted(AL),sorted(AR),sorted(IL),sorted(IR))

def restore_heads(sentence,R,AL,AR,IL,IR):
    Rmap = {i:j for (i,j) in R}
    AIR = [(i,Rmap[j]) for (i,j) in IR] + AR
    AIL = [(i,Rmap[j]) for (i,j) in IL] + AL
    head = {}
    for (i,j) in AIR:
        head[j] = i
    for (i,j) in AIL:
        head[i] = j
    for tok in sentence:
        i = tint(tok.id)
        if i in head:
            tok.head = "{}".format(head[i])

def print_statline(max_depth1,depths):
    if max_depth1 == 0:
        return
    print("# ", end="")
    for i in range(1,max_depth1+1):
        print("{:9d}".format(i), end=" ")
    print("")
    print("# ", end="")
    for i in range(1,max_depth1+1):
        print("{:9d}".format(depths[i]), end=" ")
    print("")
    sum = 0 
    print("# ", end="")
    for i in range(1,max_depth1+1):
        sum += depths[i]
        print("{:9d}".format(sum), end=" ")
    print("")
    sum = 0 
    print("# ", end="")
    for i in range(1,max_depth1+1):
        sum += depths[i]
        perc = (100.0 * sum) / all
        print("{:8.2f}%".format(perc), end=" ")
    print("")
            
def print_stat(args,sents,all,nonxes,projs,max_depth1,depths,depths_nx,depths_pj):
    if not args.nonx and not args.proj:
        print("# all graphs  "," {:5.1f}%  {:7d} (with ellipsis {:d})".format(100.0 * sents / sents, sents, sents-all))
        print("# external    "," {:5.1f}%  {:7d} (with crossing {:d})".format(100.0 * all / sents, all, all-nonxes))
        print_statline(max_depth1,depths)
    if not args.proj and not args.all:
        print("# noncrossing "," {:5.1f}%  {:7d} (with not weak projective {:d})".format(100.0 * nonxes / sents, nonxes, nonxes - projs ))
        print_statline(max_depth1,depths_nx)
    if not args.nonx and not args.all:
        print("# nx & w.proj "," {:5.1f}%  {:7d}".format(100.0 * projs / sents, projs))
        print_statline(max_depth1,depths_pj)
    print("# ")

def trace_options(sentence,args):
    value = ""
    args_list = "{}".format(args)[10:-1].split(", ")
    for arg in args_list:
        (attr,val) = arg.split("=")
        if val=='1':
            value += " --" + attr
    sentence.set_meta("command_line_options", value)

sents = bads = all = nonxes = projs = max_depth1 = max_depth2 = 0
depths, depths_nx, depths_pj = [0]*21, [0]*21, [0]*21

if (args.prop or args.trace or args.string) and not args.conll:
    print("error: add --conll to see string, properties and command line options", file=sys.stderr)
    exit(1)

for f in args.filename:
    if f == '-':
        f = '/dev/stdin'
    corpus = pyconll.load_from_file(f)
    for sentence in corpus:
        sents += 1
        if is_bad(sentence):
            bads += 1
            continue
        all += 1

        if args.trace:
            trace_options(sentence,args)

        if args.decode:
            if not has_supertags(sentence):
                print("error: no supertags", file=sys.stderr)
                exit(1)
            codestr = getcodestr(sentence)
            (Vn,R,AL,AR,IL,IR) = decode(codestr)
            (Vn,A) = undo_rope_decomposition(Vn,R,AL,AR,IL,IR)
            restore_heads(sentence,R,AL,AR,IL,IR)
        elif not has_supertags(sentence):
            (Vn,A) = digraph(sentence)
        else:
            print("error: you can only decode the input ({})".format(f), file=sys.stderr)
            exit(1)
        if args.tests:
            saved_sentence_conll = sentence.conll()

        # TESTS: noncrossing and projective
        if args.prop or args.stat or args.nonx or args.proj or args.tests:
            (verinonx,nonxfailure) = is_nonx(A)
        if args.prop or args.stat or args.proj:
            (veriproj,projfailure) = is_nonx_wproj_with_root(sentence,A,verinonx)
        # note that projetivity does not imply treeness (see Yli-Jyrä 2005)!

        if args.stat or args.prop or not args.decode:
            # ENCODING
            (Vn,E,Elabel) = underlying_graph(sentence)
            (Vn,R,AL,AR,IL,IR) = rope_decomposition(Vn, proper_rope_cover(E), A)
            (codestr,depth1,depth2,nonx) = transitions(Vn,R,AL,AR,IL,IR,Elabel)
            if args.tests:
                assert(nonx == verinonx)
                assert(codestr == getcodestr(sentence))
            # STATISTICS
            if args.stat:
                max_depth1 = max(depth1, max_depth1)
                max_depth2 = max(depth2, max_depth2)
                depths[depth1] += 1
                if verinonx:
                    depths_nx[depth1] += 1
                    nonxes += 1
                if verinonx and veriproj:
                    depths_pj[depth1] += 1
                    projs += 1

        if args.prop:
            # COMPLEXITY 
            sentence.set_meta("stack_1_size","{}".format(depth1))
            sentence.set_meta("stack_2_size","{}".format(depth2))
            value = ""
            if verinonx:
                value = "noncrossing, "
            else:
                value = "crossing{}, ".format(nonxfailure)
            if veriproj:
                value += "projective "
            else:
                value += "not projective{}".format(projfailure)
            sentence.set_meta("properties",value)
        if args.string:
            sentence.set_meta("codestring","".join(codestr))
        if args.conll:
            # FILTERED CONLL OUTPUT
            if (not args.nonx or verinonx) and (not args.proj or veriproj):
                # REDUCTION
                if not args.copy:
                    if args.decode:
                        rm_supertags(sentence)
                    else:
                        rm_heads(sentence)
                print(sentence.conll())
                print("")

        if args.tests:
            (Vnx,Rx,ALx,ARx,ILx,IRx) = decode(codestr)
            assert(Vn == Vnx)
            assert(R == Rx)
            assert(AL == ALx)
            assert(AR == ARx)
            assert(IL == ILx)
            assert(IR == IRx)
            restore_heads(sentence,Rx,ALx,ARx,ILx,IRx)
            rm_supertags(sentence)
            assert(sentence.conll() == saved_sentence_conll)

if args.stat:
    print_stat(args,sents,all,nonxes,projs,max_depth1,depths,depths_nx,depths_pj)

