# User's guide for conllenc.py v.1 

"supertag encoder and decoder of dependency graphs" 
- currently restricted to (primary) dependencies in CoNLL-U files
- converts between the HEAD column and SuperTags in MISC column 
- based on depconv v.0.1 (c) 2015-2019 Anssi Yli-Jyrä (and Carlos Gómez-Rodríguez)
- supporting also graph property annotation, codestring extraction and statistics
   
## Usage: 

       conllenc.py [-h] [--trace] [--decode] [--copy] [--indices] [--deprel]
                   [--pos] [--conll] [--string] [--prop] [--stat] [--all]
                   [--nonx] [--proj] [--tests] [--version]
                   [filename [filename ...]]

## Positional arguments:

       filename    The name of the file to transform

## Optional arguments:

- -h, --help  show this help message and exit
- --trace     Print command line switches in the beginning
- --decode    Decode heads from MISC field
- --copy      Keep both supertags and heads
- --indices   Add edge indices to printed brackets
- --deprel    Print also the arc label
- --pos       Print also the POS tag
- --conll     Produce the CoNLL format
- --string    Produce the one-line encoding format
- --prop      Print computed properties for each graph
- --stat      Print statistics at the end
- --all       Do not print stats anything but all graphs
- --nonx      Do not print stats anything but noncrossing graphs
- --proj      Do not print stats anything but projective graphs
- --tests     Run implicit unit tests
- --version   Print version number

## Typical usage:

     conllenc.py          en-ud-dev.conllu           --conll > en-ud-dev.supertags.conllu
     
     conllenc.py --decode en-ud-dev.supertags.conllu --conll > en-ud-dev.restored.conllu
     
     conllenc.py --stat   en-ud-dev.conllu
     
     conllenc.py          en-ud-dev.conllu --string  --conll | egrep '^$|# codestring = '

## Typical files

### input file:

     1	From	from	ADP	IN	_	3	case	_	_
     2	the	the	DET	DT	-	3	det	_	_
     3	AP	AP	PROPN	NNP	-	4	obl	_	_
     4	comes	come	VERB	VBZ	-	0	root	_	_
     5	this	this	DET	DT	-	6	det	_	_
     6	story	story	NOUN	NN	-	4	nsubj	_	_
     7	:	:	PUNCT	:	_	4	punct	_	_

### Typical output with --conll:

     1	From	from	ADP	IN	_	0	case	_	SuperTag⟦
     2	the	the	DET	DT	_	0	det	_	SuperTag]⁰[<
     3	AP	AP	PROPN	NNP	_	0	obl	_	SuperTag⟧⟦
     4	comes	come	VERB	VBZ	_	0	root	_	SuperTag⟧⟦
     5	this	this	DET	DT	_	0	det	_	SuperTag⟦
     6	story	story	NOUN	NN	_	0	nsubj	_	SuperTag⟧]>[⁰
     7	:	:	PUNCT	:	_	0	punct	_	SuperTag⟧>

### Typical output with --conll --indices --deprel:

     1	From	from	ADP	IN	_	0	case	_	SuperTag⟦
     2	the	the	DET	DT	_	0	det	_	SuperTag]⁰[<det(2,3)
     3	AP	AP	PROPN	NNP	_	0	obl	_	SuperTag⟧case(1,3)⟦
     4	comes	come	VERB	VBZ	_	0	root	_	SuperTag⟧obl(3,4)⟦
     5	this	this	DET	DT	_	0	det	_	SuperTag⟦
     6	story	story	NOUN	NN	_	0	nsubj	_	SuperTag⟧det(5,6)]>nsubj(4,6)[⁰
     7	:	:	PUNCT	:	_	0	punct	_	SuperTag⟧>punct(4,7)

### Typical output with --conll --string and egrep post-filtering and with --deprel --pos --indices options:

     codestring =  _From ⟦· ]⁰ _the [< · ⟧ _AP ⟦· ⟧ _comes ⟦· _this ⟦· ⟧ ]> _story [⁰ · ⟧> _:

     codestring =  ADP.IN_From ⟦· ]⁰ DET.DT_the [<det(2,3) · ⟧case(1,3) PROPN.NNP_AP ⟦· ⟧obl(3,4) 
     VERB.VBZ_comes ⟦· DET.DT_this ⟦· ⟧det(5,6) ]>nsubj(4,6) NOUN.NN_story [⁰ · ⟧>punct(4,7) PUNCT.:_:

### Combining options:

- enrich the input with codestring, properties, supertags and commandline options:

      conllenc.py --conll --string --prop --copy --trace en_lines-ud-dev.conllu 

-  get subsets with --nonx --proj

      conllenc.py --conll --string --prop --copy --trace en_lines-ud-dev.conllu --proj

- print all statistics:

      conllenc.py --stat en_lines-ud-dev.conllu 

- get less statistics with --all --nonx --proj

- move head links to supertags:

      conllenc.py --conll en_lines-ud-dev.conllu > supertags.conllu

- get richer supertags with --pos --deprel --indices

      conllenc.py --conll en_lines-ud-dev.conllu --deprel --ind

- and back to head links:

      cat supertags.conllu | conllenc.py - --decode --conll 
      cat supertags.conllu | conllenc.py - --decode --conll | diff - en_lines-ud-dev.conllu
   
- run additional internal unit tests with --tests 

### Limitations:

- the CoNLL input does not handle secondary dependencies (fix intended)
- the CoNLL input does not handle ellipsis tokens (lacking precise specifications)
- the CoNLL input does not handle semantic graph banks (fix in parallel versions)
- the one-line encoding does not contain all information (lacking motivation)

### Copyright 

(c) 2019 Anssi Yli-Jyrä

### License

Apache License
