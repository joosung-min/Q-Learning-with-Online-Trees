cdef class ORF_DQN: 
    cdef int n_action
    cdef bool isFit
    cdef int maxTrees
    
    def __init__(self, n_state, n_action, replay_size, ORFparams):
        self.n_action = n_action
        self.a_model = {} # to build RF for each action
        self.a_params = {a: ORFparams for a in range(n_action)}
        self.isFit = False
        self.maxTrees = ORFparams['maxTrees']
    
    cpdef predict(self, s):            
        # s: (4,) array (for cartpole)
        
        preds = [self.a_model[a].predict(s) for a in range(self.n_action)]
        # print(preds)
        return preds

    cpdef gen_epsilon_greedy_policy(self, float epsilon, int n_action):
        
        
        def policy_function(state):
            cdef str ran
            # state: (4,) array
            ran = "_"
            q_values =[0.0, 0.0]
            if np.random.uniform(0,1) < epsilon:
                ran = "Random"
                return([random.randint(0, n_action - 1), ran, q_values])
            else:
                if self.isFit == True:
                    ran = "Model"
                    q_values = self.predict(state) # (1,2) array
                    # print(q_values)
                else: 
                    ran = "Random_notFit"
                    return([random.randint(0, n_action - 1), ran, q_values])
                    # print("passed random.randint")
            return([np.argmax(q_values), ran, q_values])# int
        
        return policy_function


    # def expandForest(self, memory):
    #     for a in range(self.n_action):
    #         self.a_params[a]['xrng'] = ORF.dataRange([v[0] for v in memory if v[1] == a])
    #         lenFor = len(self.a_model[a].forest)
    #         for i in range(lenFor+1, self.maxTrees):
    #             self.a_model[a].forest[i] = ORF.ORT(self.a_params[a]) # build new empty trees
    

    cdef replay(self, memory, int replay_size, float gamma, int episode):
        cdef int lenFor
        if len(memory) == replay_size: # Build initial Forests
            
            for a in range(self.n_action):
                self.a_params[a]['xrng'] = ORF.dataRange([v[0] for v in memory if v[1] == a])
                self.a_model[a] = ORF.ORF(self.a_params[a]) # Fit initial RFs for each action            

        if len(memory) >= replay_size: # When the memory size exceeds the replay_size, start updating the RFs            
            
            replay_data = random.sample(memory, replay_size) # replay_data consists of [state, action, next_state, reward, is_done]
            for state, action, next_state, reward, is_done in replay_data:
                
                q_values = self.predict(state) # (, n_actions)
                q_values[action] = reward + gamma * np.max(self.predict(next_state)) if is_done == False else -1000 * reward
                
                # Update the RF for the action taken
                xrng = ORF.dataRange([v[0] for v in replay_data if v[1] == action])
                self.a_model[action].update(state, q_values[action], xrng)    
            self.isFit = True
               
        if episode == 100: # expand the number of trees at episode 100            
            # expandForest(memory)
            for a in range(self.n_action):
                self.a_params[a]['xrng'] = ORF.dataRange([v[0] for v in memory if v[1] == a])
                lenFor = len(self.a_model[a].forest)
                for i in range(lenFor+1, self.maxTrees):
                    self.a_model[a].forest[i] = ORF.ORT(self.a_params[a]) # build new empty trees