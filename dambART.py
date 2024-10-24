import copy
import time
import numpy as np
import torch as trc
import json


HEADIDX = 0
TAILIDX = 1
RELATIONIDX = 2


#Field names of the network
HEAD = 'head'
TAIL = 'tail'
RELATION = 'relation'

#Configuration for Top-Down Completion
REPLACE = 0
MERGE_W_ONE_MAX = 1
MERGE_W_ALL_MAX = 2
MERGE_W_DOMINANT = 3

#Configuration for field (head/tail) updates for every iteration of prediction
NO_FUPDATE = 0
MERGE_HEADTAIL = 1
CONDMERGE_HEADTAIL = 2


PLAINTRIPLES = 0
SRTREETRIPLES = 1
DTREETRIPLES = 2

DICTMETHOD = 0
ARRAYMETHOD = 1
TENSORMETHOD = 2

class dambART:
    def __init__(self, alpha=None, gamma=None, triples=None, entities=None, relations=None, opmode=REPLACE, htmode=NO_FUPDATE, actfactor=False, compete=False, toplimit=0, downlimit=0, iter=0, relreset=False, tmode=PLAINTRIPLES, cmode=ARRAYMETHOD):
        self._setmodeconfig(opmode=opmode, htmode=htmode, cmode=cmode)
        self._setmetaparam(actfactor=actfactor, compete=compete, toplimit=toplimit, downlimit=downlimit, iter=iter, relreset=relreset)
        self._sethyparameter(alpha=alpha, gamma=gamma)

        if triples != None:
            self.tupNetConvert(triples=triples, tmode=tmode)
        elif entities != None and relations != None:
            self.nodeidx = entities
            self.relidx = relations
            self.nodeenum = {lbl:i for i,lbl in enumerate(self.nodeidx)}
            self.relenum = {lbl:i for i,lbl in enumerate(self.relidx)} 

    def _sethyparameter(self, alpha=None, gamma=None):
        if alpha != None:
            self.alpha = alpha
        if gamma != None:
            self.gamma = gamma

    def _setmodeconfig(self, opmode=None, htmode=None, cmode=None):
        if opmode != None:
            self.opmode = opmode
        if htmode != None:
            self.htmode = htmode
        if cmode != None:
            self.cmode = cmode

    def _setmetaparam(self, actfactor=None, compete=None, toplimit=None, downlimit=None, iter=None, relreset=None):
        if actfactor != None:
            self.actfactor = actfactor
        if compete != None:
            self.compete = compete
        if toplimit != None:
            self.toplimit = toplimit
        if downlimit != None:
            self.downlimit = downlimit
        if iter != None:
            self.iter = iter
        if relreset != None:
            self.relreset = relreset

    def _setnoderelindexes(self, entities=None, relations=None):
        self.nodeidx = entities
        self.relidx = relations
        self.nodeenum = {lbl:i for i,lbl in enumerate(self.nodeidx)}
        self.relenum = {lbl:i for i,lbl in enumerate(self.relidx)}

    def _genindexedtriples(self, gmodel=None):
        trplist = []
        nidx = []
        ridx = []
        nodeset = set()
        relset = set()
        for j in range(len(gmodel)):
            hd = -1
            tl = -1
            rl = -1
            hd = self.nodeenum[[lbl for lbl in gmodel[j][HEAD]][0]]
            nodeset.add(hd)
            tl = self.nodeenum[[lbl for lbl in gmodel[j][TAIL]][0]]
            nodeset.add(tl)
            rl = self.relenum[[lbl for lbl in gmodel[j][RELATION]][0]]
            relset.add(rl)
            #trplist.append([hd,tl,rl])
            trplist.append([hd,rl,tl])
        return trplist, list(nodeset), list(relset)
            

    def trip2wghtdlist(self,gmodel=None):
        
        whd = np.zeros((len(self.graphmodel),len(self.nodeenum)), dtype=np.float32)
        wtl = np.zeros((len(self.graphmodel),len(self.nodeenum)), dtype=np.float32)
        wrl = np.zeros((len(self.graphmodel),len(self.relenum)), dtype=np.float32)
        for j in range(len(gmodel)):
            for lbl in gmodel[j][HEAD]:
                idxlbl = self.nodeenum[lbl]
                whd[j][idxlbl] = gmodel[j][HEAD][lbl]
            for lbl in gmodel[j][TAIL]:
                idxlbl = self.nodeenum[lbl]
                wtl[j][idxlbl] = gmodel[j][TAIL][lbl]
            for lbl in gmodel[j][RELATION]:
                idxlbl = self.relenum[lbl]
                wrl[j][idxlbl] = gmodel[j][RELATION][lbl]

        return [whd, wtl, wrl]
    
    def trip2wghttensorlist(self, gmodel=None):
        #whd = np.zeros((len(self.graphmodel),len(self.nodeenum)), dtype=np.float32)
        #wtl = np.zeros((len(self.graphmodel),len(self.nodeenum)), dtype=np.float32)
        #wrl = np.zeros((len(self.graphmodel),len(self.relenum)), dtype=np.float32)
        whd = trc.zeros((len(self.graphmodel),len(self.nodeenum)))
        wtl = trc.zeros((len(self.graphmodel),len(self.nodeenum)))
        wrl = trc.zeros((len(self.graphmodel),len(self.relenum)))
        for j in range(len(gmodel)):
            for lbl in gmodel[j][HEAD]:
                idxlbl = self.nodeenum[lbl]
                whd[j][idxlbl] = gmodel[j][HEAD][lbl]
            for lbl in gmodel[j][TAIL]:
                idxlbl = self.nodeenum[lbl]
                wtl[j][idxlbl] = gmodel[j][TAIL][lbl]
            for lbl in gmodel[j][RELATION]:
                idxlbl = self.relenum[lbl]
                wrl[j][idxlbl] = gmodel[j][RELATION][lbl]

        return [whd, wtl, wrl]


    def tupNetConvert(self, triples=None, exclist=[], initW=1.0, tmode=PLAINTRIPLES):
        self.graphmodel = []
        self.nodeset = set()
        self.relset = set()

        self.headset = set()
        self.tailset = set()
        self.relset = set()

        self.headsetidx = {}
        self.tailsetidx = {}
        
        self.headsetstartidx = {}
        self.tailsetstartidx = {}

        self.nodeidx = []
        self.relidx = []
        self.nodeenum = {}
        self.relenum = {}
        if triples:
            if tmode == PLAINTRIPLES:
                self.trainTriplesModel(triples=triples, exclist=exclist, initW=initW)
            elif tmode == DTREETRIPLES:
                self.trainDTriplesModel(triples=triples, exclist=exclist, initW=initW)
        self.nodeidx = list(self.nodeset)
        self.relidx = list(self.relset)
        self.nodeenum = {lbl:i for i,lbl in enumerate(self.nodeidx)}
        self.relenum = {lbl:i for i,lbl in enumerate(self.relidx)}
        #w = [[j['weights'][k] for j in self.codes] for k in range(len(a))]
        if self.cmode == ARRAYMETHOD:
            self.weightmtx = self.trip2wghtdlist(gmodel=self.graphmodel)
        elif self.cmode == TENSORMETHOD:
            self.weightmtx = self.trip2wghttensorlist(gmodel=self.graphmodel)

    def trainTriplesModel(self, triples=None, exclist=[], initW=1.0):
        for ti in range(len(triples)):
        #for trp in triples:
            trp = triples[ti]
            triple = {HEAD:{},
                    TAIL:{},
                    RELATION:{}}
            if not trp[0] in exclist:
                triple[HEAD][trp[0]]=initW
                self.nodeset.add(trp[0])
                #self.headset.add(trp[0])
                if trp[0] in self.headsetidx:
                    self.headsetidx[trp[0]].add(ti)
                else:
                    self.headsetidx[trp[0]] = set([ti])
            if not trp[1] in exclist:
                triple[RELATION][trp[1]]=initW
                self.relset.add(trp[1])
                #self.relset.add(trp[1])
            if not trp[2] in exclist:
                triple[TAIL][trp[2]]=initW
                self.nodeset.add(trp[2])
                #self.tailset.add(trp[2])
                if trp[2] in self.tailsetidx:
                    self.tailsetidx[trp[2]].add(ti)
                else:
                    self.tailsetidx[trp[2]] = set([ti])
            self.graphmodel.append(triple)

    def trainDTriplesModel(self, triples=None, exclist=[], initW=1.0):
        for ti in range(len(triples)):
            trp = triples[ti]
            hdappend = True
            tlappend = True
            tripleh = {HEAD:{},
                    TAIL:{},
                    RELATION:{}}
            triplet = copy.deepcopy(tripleh)        
            if trp[0] in self.headsetstartidx:
                if not trp[1] in exclist:
                    #print(f'index first head {self.headsetstartidx[trp[0]]}, trp[0] {trp[0]} headsetstartidx size {len(self.headsetstartidx)} graphmodel size {len(self.graphmodel)}')
                    self.graphmodel[self.headsetstartidx[trp[0]]][RELATION][trp[1]] = initW
                    self.relset.add(trp[1])
                if not trp[2] in exclist:
                    self.graphmodel[self.headsetstartidx[trp[0]]][TAIL][trp[2]] = initW
                    self.nodeset.add(trp[2])
                    if trp[2] in self.tailsetidx:
                        self.tailsetidx[trp[2]].add(ti)
                    else:
                        self.tailsetidx[trp[2]] = set([ti])
                hdappend = False
            if trp[2] in self.tailsetstartidx:
                if not trp[1] in exclist:
                    #print(f'index first tail {self.tailsetstartidx[trp[2]]}, trp[2] {trp[2]} tailsetstartidx size {len(self.tailsetstartidx)} graphmodel size {len(self.graphmodel)}')
                    self.graphmodel[self.tailsetstartidx[trp[2]]][RELATION][trp[1]] = initW
                    self.relset.add(trp[1])
                if not trp[0] in exclist:
                    self.graphmodel[self.tailsetstartidx[trp[2]]][HEAD][trp[0]] = initW
                    self.nodeset.add(trp[0])
                    if trp[0] in self.headsetidx:
                        self.headsetidx[trp[0]].add(ti)
                    else:
                        self.headsetidx[trp[0]] = set([ti])
                tlappend = False
            #hidx = ti
            #tidx = ti
            hidx = len(self.graphmodel)
            tidx = len(self.graphmodel)

            if hdappend:
                tidx += 1
                if not trp[0] in exclist:
                    tripleh[HEAD][trp[0]]=initW
                    self.nodeset.add(trp[0])
                    if trp[0] in self.headsetidx:
                        self.headsetidx[trp[0]].add(hidx)
                    else:
                        self.headsetidx[trp[0]] = set([hidx])
                if not trp[1] in exclist:
                    tripleh[RELATION][trp[1]]=initW
                    self.relset.add(trp[1])    
                if not trp[2] in exclist:
                    tripleh[TAIL][trp[2]]=initW
                    self.nodeset.add(trp[2])
                    #self.tailset.add(trp[2])
                    if trp[2] in self.tailsetidx:
                        self.tailsetidx[trp[2]].add(hidx)
                    else:
                        self.tailsetidx[trp[2]] = set([hidx])
                #print(f'headsetstartidx trp[0] {trp[0]} hidx {hidx}')
                self.headsetstartidx[trp[0]] = hidx
                self.graphmodel.append(tripleh)

            if tlappend:
                if not trp[0] in exclist:
                    triplet[HEAD][trp[0]]=initW
                    self.nodeset.add(trp[0])
                    if trp[0] in self.headsetidx:
                        self.headsetidx[trp[0]].add(hidx)
                    else:
                        self.headsetidx[trp[0]] = set([hidx])
                if not trp[1] in exclist:
                    triplet[RELATION][trp[1]]=initW
                    self.relset.add(trp[1])    
                if not trp[2] in exclist:
                    triplet[TAIL][trp[2]]=initW
                    self.nodeset.add(trp[2])
                    #self.tailset.add(trp[2])
                    if trp[2] in self.tailsetidx:
                        self.tailsetidx[trp[2]].add(hidx)
                    else:
                        self.tailsetidx[trp[2]] = set([hidx])
                #print(f'tailsetstartidx trp[2] {trp[2]} tidx {tidx}')
                self.tailsetstartidx[trp[2]] = tidx
                self.graphmodel.append(triplet)

    

    def encodeTupConvert(self, triples=None, initW=1.0, tmode=PLAINTRIPLES):
        #self.graphmodel = []
        #if triples:
        #    for trp in triples:
        #        triple = {HEAD:{},
        #                TAIL:{},
        #                RELATION:{}}
        #        triple[HEAD][trp[0]]=initW
        #        triple[RELATION][trp[1]]=initW
        #        triple[TAIL][trp[2]]=initW
        #        self.graphmodel.append(triple)

        self.graphmodel = []
        self.nodeset = set()
        self.relset = set()

        self.headset = set()
        self.tailset = set()
        self.relset = set()

        self.headsetidx = {}
        self.tailsetidx = {}
        
        self.headsetstartidx = {}
        self.tailsetstartidx = {}

        if triples:
            if tmode == PLAINTRIPLES:
                self.trainTriplesModel(triples=triples, initW=initW)
            elif tmode == DTREETRIPLES:
                self.trainDTriplesModel(triples=triples, initW=initW)
        #self.nodeidx = list(self.nodeset)
        #self.relidx = list(self.relset)
        #self.nodeenum = {lbl:i for i,lbl in enumerate(self.nodeidx)}
        #self.relenum = {lbl:i for i,lbl in enumerate(self.relidx)}
        if self.cmode == ARRAYMETHOD:
            self.weightmtx = self.trip2wghtdlist(gmodel=self.graphmodel)
        if self.cmode == TENSORMETHOD:
            self.weightmtx = self.trip2wghttensorlist(gmodel=self.graphmodel)
        


    def tupNetConvertIdx(self, triples=None, exclist=[], initW=1.0):
        nodeset = set()
        relset = set()
        if triples:
            for trp in triples:
                triple = {HEAD:{},
                        TAIL:{},
                        RELATION:{}}
                if not trp[0] in exclist:
                    triple[HEAD][trp[0]]=initW
                    nodeset.add(trp[0])
                if not trp[1] in exclist:
                    triple[RELATION][trp[1]]=initW
                    relset.add(trp[1])
                if not trp[2] in exclist:
                    triple[TAIL][trp[2]]=initW
                    nodeset.add(trp[2])

    def choiceFieldDict(self, xk=None, wjk=None, alphak=None, gammak=None):
        act = 0
        noml = list(set(xk.keys())&set(wjk.keys()))
        if len(noml) > 0:
            nom = sum([min(xk[nm],wjk[nm]) for nm in noml])
            denom = sum(wjk.values()) + alphak
            if denom > 0:
                act = gammak * (nom/denom)
        return act

    def choiceDict(self,inputd=None):
        sgv = sum(self.gamma.values())
        acts = {}
        for code in range(len(self.graphmodel)):
            nchoice =  sum([self.choiceFieldDict(xk=inputd[k], wjk=self.graphmodel[code][k],alphak=self.alpha[k], gammak=self.gamma[k]) for k in inputd])/sgv
            if nchoice > self.toplimit:
                acts[code] = nchoice
        return acts
    
    def choiceArray(self,inputa=None):
        sgv = sum(self.gamma.values())
        #inhead = np.zeros(len(self.nodeenum),dtype=np.float32)
        #intail = np.zeros(len(self.nodeenum),dtype=np.float32)
        #inrel = np.zeros(len(self.relenum),dtype=np.float32)
        #print(f'array construction done')
        #for hi in inputd[HEAD]:
        #    inhead[self.nodeenum[hi]] = inputd[HEAD][hi]
        #for ti in inputd[TAIL]:
        #    intail[self.nodeenum[ti]] = inputd[TAIL][ti]
        #for ri in inputd[RELATION]:
        #    inrel[self.relenum[ri]] = inputd[RELATION][ri]   
        #choicehd = np.dot(inhead, np.transpose(self.weightmtx[HEADIDX]))*self.gamma[HEAD]
        #choicetl = np.dot(intail, np.transpose(self.weightmtx[TAILIDX]))*self.gamma[TAIL]
        #choicerl = np.dot(inrel, np.transpose(self.weightmtx[RELATIONIDX]))*self.gamma[RELATION]
        inhead = inputa[HEADIDX]
        intail = inputa[TAILIDX]
        inrel = inputa[RELATIONIDX]
        choicehd = np.matmul(inhead, np.transpose(self.weightmtx[HEADIDX]))*self.gamma[HEAD]
        choicetl = np.matmul(intail, np.transpose(self.weightmtx[TAILIDX]))*self.gamma[TAIL]
        choicerl = np.matmul(inrel, np.transpose(self.weightmtx[RELATIONIDX]))*self.gamma[RELATION]
        print(f'choices dot done') 
        choices = (choicehd + choicetl + choicerl)/sgv
        return choices, inhead, intail, inrel
    
    def choiceTensor(self, inputa=None):
        sgv = sum(self.gamma.values())
        inhead = inputa[HEADIDX]
        intail = inputa[TAILIDX]
        inrel = inputa[RELATIONIDX]
        choicehd = trc.matmul(inhead, trc.t(self.weightmtx[HEADIDX]))*self.gamma[HEAD]
        choicetl = trc.matmul(intail, trc.t(self.weightmtx[TAILIDX]))*self.gamma[TAIL]
        choicerl = trc.matmul(inrel, trc.t(self.weightmtx[RELATIONIDX]))*self.gamma[RELATION]
        #choicehd = trc.matmul(inhead, trc.Tensor.permute(self.weightmtx[HEADIDX]))*self.gamma[HEAD]
        #choicetl = trc.matmul(intail, trc.Tensor.permute(self.weightmtx[TAILIDX]))*self.gamma[TAIL]
        #choicerl = trc.matmul(inrel, trc.Tensor.permute(self.weightmtx[RELATIONIDX]))*self.gamma[RELATION]
        print(f'choices dot done') 
        choices = (choicehd + choicetl + choicerl)/sgv
        return choices, inhead, intail, inrel

    def retrieveDict(self,J,factor=1.0):
        ret = {}
        if J < len(self.graphmodel):
            ret = {k:{lbl:ftd for lbl in self.graphmodel[J][k] for ftd in [self.graphmodel[J][k][lbl]*factor] } for k in self.graphmodel[0] }
        return ret
    
    def mergeFields(self, src=None, target=None, MaxOp=True):
        ret = copy.deepcopy(target)
        for k in src:
            if MaxOp:
                for i in src[k]:
                    if i in ret[k]:
                        ret[k][i] = max(src[k][i], ret[k][i])
                    else:
                        ret[k][i] = src[k][i]
            else:
                retmp = {}
                noml = list(set(src[k].keys())&set(target[k].keys()))
                for i in noml:
                    retmp[i] = min(src[k][i], ret[k][i])
                ret[k] = retmp
        return ret
    
    def cleanSrcAttr(self, src=None, target=None):
        ret = copy.deepcopy(target)
        for k in src:
            if k in ret:
                for lbl in src[k]:
                    if lbl in ret[k]:
                        ret[k].pop(lbl)
        return ret
    
    def selectActivationsArray(self,inputd=None):
        start = time.time()
        acts, inhd, intl, inrl = self.choiceArray(inputa=inputd)
        if self.compete:
            mx = np.max(acts)
            Jv = np.zeros(len(acts))
            for i in range(len(acts)):
                if acts[i] > self.toplimit and acts[i] >= mx:
                    if self.actfactor:
                        Jv[i] = acts[i]
                    else:
                        Jv[i] = 1.0
            print(f'selectActivation time {time.time()-start}')
            return Jv, inhd, intl, inrl
        else:
            if self.toplimit > 0:
                for i in range(len(acts)):
                    if acts[i] < self.toplimit:
                        acts[i] = 0.0
            return acts, inhd, intl, inrl
        
    def  selectActivationsTensor(self, inputd=None):
        start = time.time()
        acts, inhd, intl, inrl = self.choiceTensor(inputa=inputd)
        if self.compete:
            mx = trc.max(acts)
            Jv = trc.zeros(len(acts))
            for i in range(len(acts)):
                if acts[i] > self.toplimit and acts[i] >= mx:
                    if self.actfactor:
                        Jv[i] = acts[i]
                    else:
                        Jv[i] = 1.0
            print(f'selectActivation time {time.time()-start}')
            return Jv, inhd, intl, inrl
        else:
            if self.toplimit > 0:
                for i in range(len(acts)):
                    if acts[i] < self.toplimit:
                        acts[i] = 0.0
            return acts, inhd, intl, inrl

    def selectActivationsDict(self,inputd=None):
        start = time.time()
        acts = self.choiceDict(inputd=inputd)
        Jdict = {}
        if self.compete:
            mxk = [k for k,v in acts.items() if v >= max(acts.values())]
            if len(mxk) > 0:
                if acts[mxk[0]] > self.toplimit:
                    Jdict[mxk[0]] = acts[mxk[0]] 
        else:
            Jdict = {k:v for k,v in acts.items() if v > self.toplimit}
        print(f'{len(acts)} activations from selectActivation time {time.time()-start}')
        return Jdict

    def completionArray(self, inputd=None):
        #print(f'inputd {inputd}')
        sacts, inhd, intl, inrl = self.selectActivationsArray(inputd=inputd)
        outhd = np.zeros(len(inhd))
        outtl = np.zeros(len(intl))
        outrl = np.zeros(len(inrl))
        sumTj = 0.0
        if self.opmode == REPLACE:
            midxs = list(np.where(sacts == np.max(sacts))[0])
            for j in range(len(midxs)):
                outhd = self.weightmtx[HEADIDX][j]
                outtl = self.weightmtx[TAILIDX][j]
                outrl = self.weightmtx[RELATIONIDX][j]
        elif self.opmode == MERGE_W_ONE_MAX:
            midx = np.max(sacts)
            outhd = np.amax(np.array([self.weightmtx[HEADIDX][midx],inhd]), axis=0)
            outtl = np.amax(np.array([self.weightmtx[TAILIDX][midx],intl]), axis=0)
            outrl = np.amax(np.array([self.weightmtx[RELATIONIDX][midx],inrl]), axis=0)
        elif self.opmode == MERGE_W_ALL_MAX:
            outhd = inhd
            outtl = intl
            outrl = inrl
            midxs = list(np.where(sacts == np.max(sacts))[0])
            for j in range(len(midxs)):
                outhd = np.amax(np.array([self.weightmtx[HEADIDX][midx],outhd]), axis=0)
                outtl = np.amax(np.array([self.weightmtx[TAILIDX][midx],outtl]), axis=0)
                outrl = np.amax(np.array([self.weightmtx[RELATIONIDX][midx],outrl]), axis=0)
        elif self.opmode == MERGE_W_DOMINANT:
            sumTj = np.sum(sacts)
            self.actfactor = True
            outhd = np.matmul(sacts/sumTj, self.weightmtx[HEADIDX])
            outtl = np.matmul(sacts/sumTj, self.weightmtx[TAILIDX])
            outrl = np.matmul(sacts/sumTj, self.weightmtx[RELATIONIDX])
            #outhd = np.amax(np.array([outhd,inhd]), axis=0)
            #outtl = np.amax(np.array([outtl,intl]), axis=0)
            #outrl = np.amax(np.array([outrl,inrl]), axis=0)
        return sacts, outhd, outtl, outrl

    def completionTensor(self, inputd=None):
        sacts, inhd, intl, inrl = self.selectActivationsTensor(inputd=inputd)
        outhd = trc.zeros(len(inhd))
        outtl = trc.zeros(len(intl))
        outrl = trc.zeros(len(inrl))
        sumTj = 0.0
        if self.opmode == REPLACE:
            midxs = list(trc.where(sacts == trc.max(sacts))[0])
            for j in range(len(midxs)):
                outhd = self.weightmtx[HEADIDX][j]
                outtl = self.weightmtx[TAILIDX][j]
                outrl = self.weightmtx[RELATIONIDX][j]
        elif self.opmode == MERGE_W_ONE_MAX:
            midx = trc.max(sacts)
            #outhd = trc.amax(trc.tensor([self.weightmtx[HEADIDX][midx],inhd]), axis=0)
            #outtl = trc.amax(trc.tensor([self.weightmtx[TAILIDX][midx],intl]), axis=0)
            #outrl = trc.amax(trc.tensor([self.weightmtx[RELATIONIDX][midx],inrl]), axis=0)
            outhd = trc.amax(trc.t(trc.stack((self.weightmtx[HEADIDX][midx],inhd))),1)               
            outtl = trc.amax(trc.t(trc.stack((self.weightmtx[TAILIDX][midx],intl))),1) 
            outrl = trc.amax(trc.t(trc.stack((self.weightmtx[RELATIONIDX][midx],inrl))),1)
        elif self.opmode == MERGE_W_ALL_MAX:
            outhd = inhd
            outtl = intl
            outrl = inrl
            midxs = list(trc.where(sacts == np.max(sacts))[0])
            for j in range(len(midxs)):
                outhd = trc.amax(trc.t(trc.stack((self.weightmtx[HEADIDX][midx],outhd))),1) 
                outtl = trc.amax(trc.t(trc.stack((self.weightmtx[TAILIDX][midx],outtl))),1)
                outrl = trc.amax(trc.t(trc.stack((self.weightmtx[RELATIONIDX][midx],outrl))),1)
        elif self.opmode == MERGE_W_DOMINANT:
            sumTj = trc.sum(sacts)
            self.actfactor = True
            outhd = trc.matmul(sacts/sumTj, self.weightmtx[HEADIDX])
            outtl = trc.matmul(sacts/sumTj, self.weightmtx[TAILIDX])
            outrl = trc.matmul(sacts/sumTj, self.weightmtx[RELATIONIDX])
            #outhd = np.amax(np.array([outhd,inhd]), axis=0)
            #outtl = np.amax(np.array([outtl,intl]), axis=0)
            #outrl = np.amax(np.array([outrl,inrl]), axis=0)
        return sacts, outhd, outtl, outrl

    def completionDict(self,inputd=None, filteredfld=[]):
        sacts = self.selectActivationsDict(inputd=inputd)
        #print(f'sacts {sacts}')
        #print('ready to top-down merge')
        retin = copy.deepcopy(inputd)
        Js = {}
        sumTj = 0.0
        if self.opmode == MERGE_W_DOMINANT:
            sumTj =  sum(sacts.values())
            self.actfactor = True
        retsum = {k:{} for k in retin}
        #print(f'Starting top-down down limit {self.downlimit}')
        #print(f'self.htmode {self.htmode}')
        for idx in sacts:
            if self.actfactor:
                retrvd = self.retrieveDict(idx, factor=sacts[idx])
            else:
                retrvd = self.retrieveDict(idx)
            if self.opmode == REPLACE:
                for ridx in retin:
                    if not ridx in filteredfld:
                        retin[ridx] = retrvd[ridx]
            elif self.opmode == MERGE_W_ONE_MAX:
                for k in retrvd:
                    retrvd[k] = {lbl:retrvd[k][lbl] for lbl in retrvd[k] if retrvd[k][lbl] > self.downlimit}
                retrvtmp = self.mergeFields(src=retrvd, target=retin, MaxOp=False)
                for ridx in retrvtmp:
                    if not ridx in filteredfld:
                        retin[ridx] = retrvtmp[ridx]
            elif self.opmode == MERGE_W_ALL_MAX:
                for k in retrvd:
                    retrvd[k] = {lbl:retrvd[k][lbl] for lbl in retrvd[k] if retrvd[k][lbl] > self.downlimit}
                retrvtmp = self.mergeFields(src=retrvd, target=retin)
                for ridx in retrvtmp:
                    if not ridx in filteredfld:
                        retin[ridx] = retrvtmp[ridx]
            elif self.opmode == MERGE_W_DOMINANT:
                for k in retrvd:
                    popatt = False
                    for att in retrvd[k]:
                        if att in retsum[k]:
                            retsum[k][att] += retrvd[k][att]/sumTj
                        else:
                            retsum[k][att] = retrvd[k][att]/sumTj
                srcmerge = {k:{} for k in retsum}
                for k in retsum:
                    for lbl in retsum[k]:
                        if retsum[k][lbl] >= self.downlimit:
                            srcmerge[k][lbl] = retsum[k][lbl] 
                retrvtmp = self.mergeFields(src=srcmerge, target=retin)
                for ridx in retrvtmp:
                    if not ridx in filteredfld:
                        retin[ridx] = retrvtmp[ridx]
            Js[idx]=sacts[idx]
            if self.compete:
                break
        return Js, retin
    

    def resetFields(self,inputd=None, exceptFld=[]):
        retinput = {k:{} for k in inputd}
        for k in exceptFld:
            if k in inputd:
                retinput[k] = inputd[k]
        return retinput

    def indict2Array(self, inputd=None):
        inhead = np.zeros(len(self.nodeenum),dtype=np.float32)
        intail = np.zeros(len(self.nodeenum),dtype=np.float32)
        inrel = np.zeros(len(self.relenum),dtype=np.float32)
        for hi in inputd[HEAD]:
            inhead[self.nodeenum[hi]] = inputd[HEAD][hi]
        for ti in inputd[TAIL]:
            intail[self.nodeenum[ti]] = inputd[TAIL][ti]
        for ri in inputd[RELATION]:
            inrel[self.relenum[ri]] = inputd[RELATION][ri]
        print(f'input to array construction done')
        return inhead, intail, inrel

    def indict2tensor(self, inputd=None):
        inhead = trc.zeros(len(self.nodeenum))
        intail = trc.zeros(len(self.nodeenum))
        inrel = trc.zeros(len(self.relenum))
        for hi in inputd[HEAD]:
            inhead[self.nodeenum[hi]] = inputd[HEAD][hi]
        for ti in inputd[TAIL]:
            intail[self.nodeenum[ti]] = inputd[TAIL][ti]
        for ri in inputd[RELATION]:
            inrel[self.relenum[ri]] = inputd[RELATION][ri]
        print(f'input to tensor construction done')
        return inhead, intail, inrel


    def graphPredictionArray(self, inputd=None, cleanInput=True):
        ipredict = copy.deepcopy(inputd)
        opredict = copy.deepcopy(ipredict)
        mohidx = list(np.where(opredict[HEADIDX] == np.max(opredict[HEADIDX]))[0])
        motidx = list(np.where(opredict[TAILIDX] == np.max(opredict[TAILIDX]))[0])
        moridx = list(np.where(opredict[RELATIONIDX] == np.max(opredict[RELATIONIDX]))[0])
        for i in range(self.iter):
            print(f'cycle#{i}')
            jact, ophead, optail, oprelation = self.completionArray(inputd=ipredict)
            if self.htmode == NO_FUPDATE:
                ipredict = [copy.deepcopy(ophead), copy.deepcopy(optail), copy.deepcopy(oprelation)]
            elif self.htmode == MERGE_HEADTAIL:
                imerge = np.amax(np.array([ophead,optail]), axis=0)
                ipredict = [copy.deepcopy(imerge), copy.deepcopy(imerge), copy.deepcopy(oprelation)]
            elif self.htmode == CONDMERGE_HEADTAIL:
                ipredict = [copy.deepcopy(ophead), copy.deepcopy(optail), copy.deepcopy(oprelation)]
                imerge = np.amax(np.array([ophead,optail]), axis=0)
                if len(mohidx) < len(opredict[HEADIDX]):
                    ipredict[HEADIDX] = copy.deepcopy(imerge)
                if len(motidx) < len(opredict[TAILIDX]):
                    ipredict[TAILIDX] = copy.deepcopy(imerge)
            if self.relreset:
                ipredict[RELATIONIDX] = np.zeros(len(oprelation))
            orioutput = [ophead, optail, oprelation]
            print(f'end cycle#{i}')
        if cleanInput:
            #if len(mohidx) < len(opredict[HEADIDX]):
            #    for i in range(len(mohidx)):
            #        ipredict[HEADIDX][mohidx[i]] = 1 - opredict[HEADIDX][mohidx[i]]
            #        if self.htmode == MERGE_HEADTAIL:
            #            ipredict[TAILIDX][mohidx[i]] = 1 - opredict[HEADIDX][mohidx[i]]
            ipredict[HEADIDX] = np.amax(np.array([ophead,opredict[HEADIDX]]),axis=0) 
            #ipredict[HEADIDX] = ophead
            #if len(motidx) < len(opredict[TAILIDX]):
            #    for i in range(len(motidx)):
            #        ipredict[TAILIDX][motidx[i]] = 1 - opredict[TAILIDX][motidx[i]]
            #        if self.htmode == MERGE_HEADTAIL:
            #            ipredict[HEADIDX][mohidx[i]] = 1 - opredict[TAILIDX][mohidx[i]]
            ipredict[TAILIDX] = np.amax(np.array([optail,opredict[TAILIDX]]),axis=0)
            #ipredict[TAILIDX] = optail
            #if len(moridx) < len(opredict[RELATIONIDX]):
            #    for i in range(len(moridx)):
            #        ipredict[RELATIONIDX][moridx[i]] = 1 - opredict[RELATIONIDX][moridx[i]]
            ipredict[RELATIONIDX] = np.amax(np.array([oprelation,opredict[RELATIONIDX]]),axis=0)
            #ipredict[RELATIONIDX] = oprelation
        return jact, ipredict, opredict
        

    def graphPredictionTensor(self,inputd=None, cleanInput=True):
        ipredict = copy.deepcopy(inputd)
        opredict = copy.deepcopy(ipredict)
        mohidx = list(trc.where(opredict[HEADIDX] == trc.max(opredict[HEADIDX]))[0])
        motidx = list(trc.where(opredict[TAILIDX] == trc.max(opredict[TAILIDX]))[0])
        moridx = list(trc.where(opredict[RELATIONIDX] == trc.max(opredict[RELATIONIDX]))[0])
        for i in range(self.iter):
            print(f'cycle#{i}')
            jact, ophead, optail, oprelation = self.completionTensor(inputd=ipredict)
            if self.htmode == NO_FUPDATE:
                ipredict = [copy.deepcopy(ophead), copy.deepcopy(optail), copy.deepcopy(oprelation)]
            elif self.htmode == MERGE_HEADTAIL:
                #imerge = trc.amax(trc.tensor([ophead,optail]), axis=0)
                imerge = trc.amax(trc.t(trc.stack((ophead,optail))),1) 
                ipredict = [copy.deepcopy(imerge), copy.deepcopy(imerge), copy.deepcopy(oprelation)]
            elif self.htmode == CONDMERGE_HEADTAIL:
                ipredict = [copy.deepcopy(ophead), copy.deepcopy(optail), copy.deepcopy(oprelation)]
                #imerge = trc.amax(trc.tensor([ophead,optail]), axis=0)
                imerge = trc.amax(trc.t(trc.stack((ophead,optail))),1)
                if len(mohidx) < len(opredict[HEADIDX]):
                    ipredict[HEADIDX] = copy.deepcopy(imerge)
                if len(motidx) < len(opredict[TAILIDX]):
                    ipredict[TAILIDX] = copy.deepcopy(imerge)
            if self.relreset:
                ipredict[RELATIONIDX] = trc.zeros(len(oprelation))
            orioutput = [ophead, optail, oprelation]
            print(f'end cycle#{i}')
        if cleanInput:
            #if len(mohidx) < len(opredict[HEADIDX]):
            #    for i in range(len(mohidx)):
            #        ipredict[HEADIDX][mohidx[i]] = 1 - opredict[HEADIDX][mohidx[i]]
            #        if self.htmode == MERGE_HEADTAIL:
            #            ipredict[TAILIDX][mohidx[i]] = 1 - opredict[HEADIDX][mohidx[i]]
            #ipredict[HEADIDX] = trc.amax(trc.tensor([ophead,opredict[HEADIDX]]),axis=0) 
            ipredict[HEADIDX] = trc.amax(trc.t(trc.stack((ophead,opredict[HEADIDX]))),1)
            #ipredict[HEADIDX] = ophead
            #if len(motidx) < len(opredict[TAILIDX]):
            #    for i in range(len(motidx)):
            #        ipredict[TAILIDX][motidx[i]] = 1 - opredict[TAILIDX][motidx[i]]
            #        if self.htmode == MERGE_HEADTAIL:
            #            ipredict[HEADIDX][mohidx[i]] = 1 - opredict[TAILIDX][mohidx[i]]
            #ipredict[TAILIDX] = trc.amax(trc.tensor([optail,opredict[TAILIDX]]),axis=0)
            ipredict[TAILIDX] = trc.amax(trc.t(trc.stack((optail,opredict[TAILIDX]))),1)
            #ipredict[TAILIDX] = optail
            #if len(moridx) < len(opredict[RELATIONIDX]):
            #    for i in range(len(moridx)):
            #        ipredict[RELATIONIDX][moridx[i]] = 1 - opredict[RELATIONIDX][moridx[i]]
            #ipredict[RELATIONIDX] = trc.amax(trc.tensor([oprelation,opredict[RELATIONIDX]]),axis=0)
            ipredict[RELATIONIDX] = trc.amax(trc.t(trc.stack((oprelation,opredict[RELATIONIDX]))),1)
            #ipredict[RELATIONIDX] = oprelation
        return jact, ipredict, opredict

    def graphPrediction(self,inputd=None, cleanInput=True):
        ipredict = copy.deepcopy(inputd)
        opredict = copy.deepcopy(ipredict)
        jact = {}
        for i in range(self.iter):
            print(f'cycle#{i}')
            jact, opredict = self.completionDict(inputd=ipredict)
            if self.htmode == NO_FUPDATE:
                ipredict = copy.deepcopy(opredict)
            elif self.htmode == MERGE_HEADTAIL:
                tipredict = self.mergeFields(src={HEAD:opredict[TAIL]}, target={HEAD:opredict[HEAD]})
                ipredict[HEAD] = tipredict[HEAD]
                ipredict[TAIL] = copy.deepcopy(ipredict[HEAD])
            elif self.htmode == CONDMERGE_HEADTAIL:
                tipredict = self.mergeFields(src={HEAD:opredict[TAIL]}, target={HEAD:opredict[HEAD]})
                if len(opredict[HEAD]) > 0:
                    ipredict[HEAD] = tipredict[HEAD]
                if len(opredict[TAIL]) > 0:
                    ipredict[TAIL] = tipredict[TAIL]
            if self.relreset:
                ipredict = self.resetFields(inputd=ipredict, exceptFld=[HEAD,TAIL])
            else:
                ipredict[RELATION] = opredict[RELATION]
            print(f'end cycle#{i}')
        if cleanInput:
            if self.htmode == MERGE_HEADTAIL:
                opredict = self.cleanSrcAttr(src={HEAD:inputd[HEAD],TAIL:inputd[HEAD],RELATION:inputd[RELATION]}, target=opredict)
            opredict = self.cleanSrcAttr(src=inputd, target=opredict)
        return jact, opredict, ipredict
    
    def graphPredictionIndexed(self, lhs=None, rhs=None, rel=None, cleanInput=True):
        ind = {HEAD:{},TAIL:{},RELATION:{}}
        if lhs != None:
            #ind[HEAD][self.nodeidx[lhs]] = 1.0
            ind[HEAD][lhs] = 1.0
        if rhs != None:
            #ind[TAIL][self.nodeidx[rhs]] = 1.0
            ind[TAIL][rhs] = 1.0
        if rel != None:
            #ind[RELATION][self.relidx[rel]] = 1.0
            ind[RELATION][rel] = 1.0
        print(f'ind {ind}')
        if self.cmode == DICTMETHOD:
            js, tripleout, triplein = self.graphPrediction(inputd=ind, cleanInput=cleanInput)
            print('allocation')
            hdscore = np.zeros(len(self.nodeenum))
            tlscore = np.zeros(len(self.nodeenum))
            rlscore = np.zeros(len(self.relenum))
            print('start assignment')
            for i in tripleout[HEAD]:
                hdscore[self.nodeenum[i]] = tripleout[HEAD][i]
            for i in tripleout[TAIL]:
                tlscore[self.nodeenum[i]] = tripleout[TAIL][i]
            for i in tripleout[RELATION]:
                rlscore[self.relenum[i]] = tripleout[RELATION][i]
        elif self.cmode == ARRAYMETHOD:
            ahead, atail, arel = self.indict2Array(inputd=ind)
            inputarr = [ahead, atail, arel]
            js, tripleout, triplein = self.graphPredictionArray(inputd=inputarr, cleanInput=cleanInput)
            hdscore = tripleout[HEADIDX]
            tlscore = tripleout[TAILIDX]
            rlscore = tripleout[RELATIONIDX]
        elif self.cmode == TENSORMETHOD:
            ahead, atail, arel = self.indict2tensor(inputd=ind)
            inputarr = [ahead, atail, arel]
            js, tripleout, triplein = self.graphPredictionTensor(inputd=inputarr, cleanInput=cleanInput)
            hdscore = tripleout[HEADIDX]
            tlscore = tripleout[TAILIDX]
            rlscore = tripleout[RELATIONIDX]
        return [hdscore, rlscore, tlscore] #, ind, tripleout 
        

    #this is for testing only---------------------------------------------
    def graphPrediction_testperform(self,inputd=None, cleanInput=True):
        ipredict = copy.deepcopy(inputd)
        opredict = copy.deepcopy(ipredict)
        jact = {}
        for i in range(self.iter):
            print(f'cycle#{i}')
            if i > 2:
                start = time.time()
                sacts = self.selectActivationsDict(inputd=ipredict)
                print(f'selectActivation time {time.time()-start}')
            else:    
                start = time.time()
                jact, opredict = self.completionDict(inputd=ipredict)
                if self.htmode == NO_FUPDATE:
                    ipredict = copy.deepcopy(opredict)
                elif self.htmode == MERGE_HEADTAIL:
                    tipredict = self.mergeFields(src={HEAD:opredict[TAIL]}, target={HEAD:opredict[HEAD]})
                    ipredict[HEAD] = tipredict[HEAD]
                    ipredict[TAIL] = copy.deepcopy(ipredict[HEAD])
                if self.relreset:
                    ipredict = self.resetFields(inputd=ipredict, exceptFld=[HEAD,TAIL])
                else:
                    ipredict[RELATION] = opredict[RELATION]
                print(f'all completion time {time.time()-start}')
            print(f'end cycle#{i}')
        if cleanInput:
            if self.htmode == MERGE_HEADTAIL:
                opredict = self.cleanSrcAttr(src={HEAD:inputd[HEAD],TAIL:inputd[HEAD],RELATION:inputd[RELATION]}, target=opredict)
            opredict = self.cleanSrcAttr(src=inputd, target=opredict)
        return jact, opredict, ipredict
    

    def saveGraphModel(self, filename=None):
        outfile = open(filename, 'w')
        json.dump(self.graphmodel, outfile)

    def loadGraphModel(self, filename=None):
        fo_in = open(filename, 'r', encoding='utf-8')
        self.graphmodel = json.load(fo_in)



    


        