import random
from random import randrange
from time import time 


class Problem_Genetic(object):
#=====================================================================================================================================
# Class to represent problems to be solved by means of a general
# genetic algorithm. It includes the following attributes:
# - genes: list of possible genes in a chromosome
# - individuals_length: length of each chromosome
# - decode: method that receives the genotype (chromosome) as input and returns
#    the phenotype (solution to the original problem represented by the chromosome) 
# - fitness: method that returns the evaluation of a chromosome (acts over the
#    genotype)
# - mutation: function that implements a mutation over a chromosome
# - crossover: function that implements the crossover operator over two chromosomes
#=====================================================================================================================================
    
    def __init__(self,genes,individuals_length,decode,fitness):
        self.genes= genes
        self.individuals_length= individuals_length
        self.decode= decode
        self.fitness= fitness


    def mutation(self, chromosome, prob):
            
            def inversion_mutation(chromosome_aux):
                chromosome = chromosome_aux
                index1 = randrange(0,len(chromosome))
                index2 = randrange(index1,len(chromosome))
                chromosome_mid = chromosome[index1:index2]
                chromosome_mid.reverse()
                chromosome_result = chromosome[0:index1] + chromosome_mid + chromosome[index2:]
                
                return chromosome_result
        
            aux = []
            for _ in range(len(chromosome)):
                if random.random() < prob :
                    aux = inversion_mutation(chromosome)
            return aux

    def crossover(self,parent1, parent2):

        def process_gen_repeated(copy_child1,copy_child2):
            count1=0
            for gen1 in copy_child1[:pos]:
                repeat = 0
                repeat = copy_child1.count(gen1)
                if repeat > 1:#If need to fix repeated gen
                    count2=0
                    for gen2 in parent1[pos:]:#Choose next available gen
                        if gen2 not in copy_child1:
                            child1[count1] = parent1[pos:][count2]
                        count2+=1
                count1+=1

            count1=0
            for gen1 in copy_child2[:pos]:
                repeat = 0
                repeat = copy_child2.count(gen1)
                if repeat > 1:#If need to fix repeated gen
                    count2=0
                    for gen2 in parent2[pos:]:#Choose next available gen
                        if gen2 not in copy_child2:
                            child2[count1] = parent2[pos:][count2]
                        count2+=1
                count1+=1

            return [child1,child2]

        pos=random.randrange(1,self.individuals_length-1)
        child1 = parent1[:pos] + parent2[pos:] 
        child2 = parent2[:pos] + parent1[pos:] 
        
        return  process_gen_repeated(child1, child2)
        

def decodeTSP(chromosome):    
    lista=[]
    for i in chromosome:
        lista.append(cities.get(i))
    return lista


def penalty(chromosome):
        actual = chromosome
        value_penalty = 0
        for i in actual:
            times = 0
            times = actual.count(i) 
            if times > 1:
                value_penalty+= 100 * abs(times - len(actual))
        return value_penalty


def fitnessTSP(chromosome):
    
    def distanceTrip(index,city):
        w = distances.get(index)
        return  w[city]
        
    actualChromosome = list(chromosome)
    fitness_value = 0
    count = 0
    
    # Penalty for a city repetition inside the chromosome
    penalty_value = penalty(actualChromosome)
 
    for i in chromosome:
        if count==7:
            nextCity = actualChromosome[0]
        else:    
            temp = count+1
            nextCity = actualChromosome[temp]
         
        fitness_value+= distanceTrip(i,nextCity) + 50 * penalty_value
        count+=1
        
    return fitness_value



#========================================================== FIRST PART: GENETIC OPERATORS============================================
# Here We defined the requierements functions that the GA needs to work 
# The function receives as input:
# * problem_genetic: an instance of the class Problem_Genetic, with
#     the optimization problem that we want to solve.
# * k: number of participants on the selection tournaments.
# * opt: max or min, indicating if it is a maximization or a
#     minimization problem.
# * ngen: number of generations (halting condition)
# * size: number of individuals for each generation
# * ratio_cross: portion of the population which will be obtained by
#     means of crossovers. 
# * prob_mutate: probability that a gene mutation will take place.
#=====================================================================================================================================


def genetic_algorithm_t(Problem_Genetic,k,opt,ngen,size,ratio_cross,prob_mutate):
    
    def initial_population(Problem_Genetic,size):   
        def generate_chromosome():
            chromosome=[]
            for i in Problem_Genetic.genes:
                chromosome.append(i)
            random.shuffle(chromosome)
            return chromosome
        return [generate_chromosome() for _ in range(size)]
            
    def new_generation_t(Problem_Genetic,k,opt,population,n_parents,n_directs,prob_mutate):
        
        def tournament_selection(Problem_Genetic,population,n,k,opt):
            winners=[]

            for _ in range(int(n)):
                elements = random.sample(population,k)
                winners.append(opt(elements,key=Problem_Genetic.fitness))
            return winners
        
        def cross_parents(Problem_Genetic,parents):
            childs=[]
            for i in range(0,len(parents),2):
                childs.extend(Problem_Genetic.crossover(parents[i],parents[i+1]))
            return childs
    
        def mutate(Problem_Genetic,population,prob):
            for i in population:
                Problem_Genetic.mutation(i,prob)
            return population
                        
        directs = tournament_selection(Problem_Genetic, population, n_directs, k, opt)
        crosses = cross_parents(Problem_Genetic,
                                tournament_selection(Problem_Genetic, population, n_parents, k, opt))
        mutations = mutate(Problem_Genetic, crosses, prob_mutate)
        new_generation = directs + mutations
        
        return new_generation
    
    population = initial_population(Problem_Genetic, size)
    n_parents= round(size*ratio_cross)
    n_parents = (n_parents if n_parents%2==0 else n_parents-1)
    n_directs = int(size - n_parents)
    
    for _ in range(ngen):
        population = new_generation_t(Problem_Genetic, k, opt, population, n_parents, n_directs, prob_mutate)
    
    bestChromosome = opt(population, key = Problem_Genetic.fitness)
    print("Chromosome: ", bestChromosome)
    genotype = Problem_Genetic.decode(bestChromosome)
    print ("Solution:" , (genotype,Problem_Genetic.fitness(bestChromosome)))
    return (genotype,Problem_Genetic.fitness(bestChromosome))



#============================================================ SECOND PART: VARIANTS OVER THE STANDARD GENETIC ALGORITHM =========================================
# Modify the standard version of genetic algorithms developed in the previous step, by choosing only one of the following:
# Genetic Algorithm with Varying Population Size
#
# *** -> We choose this option
#
# The idea is to introduce the concept of "ageing" into the population of chromosomes. 
# Each individual will get a "life-expectancy" value, which directly depends on the fitness. Parents are selected randomly, 
# without paying attention to their fitness, but at each step all chromosomes gain +1 to their age,
# and those reaching their life-expectancy are removed from the population. 
# It is very important to design a good function calculating life-expectancy, so that better individuals survive during more generations,
# and therefore get more chances to be selected for crossover.
#
# Cellular Genetic Algorithm
# The idea is to introduce the concept of "neighbourhood" into the population of chromosomes (for instance, placing them into 
# a grid-like arrangement),
# in such a way that each individual can only perform crossover with its direct neighbours.
#=====================================================================================================================================


def genetic_algorithm_t2(Problem_Genetic,k,opt,ngen,size,ratio_cross,prob_mutate,dictionary):
    
    def initial_population(Problem_Genetic,size):  
        
        def generate_chromosome():
            chromosome=[]
            for i in Problem_Genetic.genes:
                chromosome.append(i)
            random.shuffle(chromosome)
            #Adding to dictionary new generation
            dictionary[str(chromosome)]=1
            return chromosome
        
        return [generate_chromosome() for _ in range(size)]
            
    def new_generation_t(Problem_Genetic,k,opt,population,n_parents,n_directs,prob_mutate):
        def tournament_selection(Problem_Genetic,population,n,k,opt):
            winners=[]
            for _ in range(int(n)):
                elements = random.sample(population,k)
                winners.append(opt(elements,key=Problem_Genetic.fitness))
            for winner in winners:
                #For each winner, if exists in dictionary, we increase his age
                if str(winner) in dictionary:
                    dictionary[str(winner)]=dictionary[str(winner)]+1
                #Else we need to inicializate in dictionary
                else:
                    dictionary[str(winner)]=1
            return winners
        
        def cross_parents(Problem_Genetic,parents):
            childs=[]
            for i in range(0,len(parents),2):
                childs.extend(Problem_Genetic.crossover(parents[i],parents[i+1]))
                #Each time that some parent are crossed we add their two sons to dictionary 
                if str(parents[i]) not in dictionary:
                    dictionary[str(parents[i])]=1
                dictionary[str(childs[i])]=dictionary[str(parents[i])]
                #...and remove their parents
                del dictionary[str(parents[i])]

            return childs
    
        def mutate(Problem_Genetic,population,prob):
            j = 0
            copy_population=population
            for crom in population:
                Problem_Genetic.mutation(crom,prob)
                #Each time that some parent is crossed
                if str(crom) in dictionary:
                    #We add the new chromosome mutated
                    dictionary[str(population[j])]=dictionary[str(crom)]
                    #Then we remove the parent, because his mutated has been added.
                    del dictionary[str(copy_population[j])]
                    j+=j
            return population
        
        directs = tournament_selection(Problem_Genetic, population, n_directs, k, opt)
        crosses = cross_parents(Problem_Genetic,tournament_selection(Problem_Genetic, population, n_parents, k, opt))
        mutations = mutate(Problem_Genetic, crosses, prob_mutate)
        new_generation = directs + mutations
        #Adding new generation of mutants to dictionary.
        for ind in new_generation:
            age = 0
            if str(ind) in dictionary:
                age+=1
                dictionary[str(ind)]+=1
            else:
                dictionary[str(ind)]=1
        return new_generation
  
    population = initial_population(Problem_Genetic, size )
    n_parents= round(size*ratio_cross)
    n_parents = (n_parents if n_parents%2==0 else n_parents-1)
    n_directs = size - n_parents
    
    for _ in range(ngen):
        population = new_generation_t(Problem_Genetic, k, opt, population, n_parents, n_directs, prob_mutate)
        
    bestChromosome = opt(population, key = Problem_Genetic.fitness)
    print("Chromosome: ", bestChromosome)
    genotype = Problem_Genetic.decode(bestChromosome)
    print ("Solution:" , (genotype,Problem_Genetic.fitness(bestChromosome)),dictionary[(str(bestChromosome))] ," generations of winners parents.")

    return (genotype,Problem_Genetic.fitness(bestChromosome)
            + dictionary[(str(bestChromosome))]*50) #Updating fitness with age too
    
    
    
 
#========================================================================THIRD PART: EXPERIMENTATION=========================================================
# Run over the same instances both the standard GA (from first part) as well as the modified version (from second part).
# Compare the quality of their results and their performance. Due to the inherent randomness of GA, 
# the experiments performed over each instance should be run several times.
#============================================================================================================================================================


def TSP(k):
    TSP_PROBLEM = Problem_Genetic([0,1,2,3,4,5,6,7],len(cities), lambda x : decodeTSP(x), lambda y: fitnessTSP(y))
    
    def first_part_GA(k):
        cont  = 0
        print ("---------------------------------------------------------Executing FIRST PART: TSP --------------------------------------------------------- \n")
        tiempo_inicial_t2 = time()
        while cont <= k: 
            genetic_algorithm_t(TSP_PROBLEM, 2, min, 200, 100, 0.8, 0.05)
            cont+=1
        tiempo_final_t2 = time() 
        print("")
        print("Total time: ",(tiempo_final_t2 - tiempo_inicial_t2)," secs.\n")
    
    def second_part_GA(k):
        print ("---------------------------------------------------------Executing SECOND PART: TSP --------------------------------------------------------- \n")
        cont = 0
        tiempo_inicial_t2 = time()
        while cont <= k: 
            genetic_algorithm_t2(TSP_PROBLEM, 2, min, 200, 100, 0.8, 0.05,{})
            cont+=1
        tiempo_final_t2 = time()
        print("") 
        print("Total time: ",(tiempo_final_t2 - tiempo_inicial_t2)," secs.\n")
    
    #first_part_GA(k)
    
    first_part_GA(k)
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    second_part_GA(k)

#---------------------------------------- AUXILIARY DATA FOR TESTING --------------------------------


cities = {0:'Almeria',1:'Cadiz',2:'Cordoba',3:'Granada',4:'Huelva',5:'Jaen',6:'Malaga',7:'Sevilla'}

#Distance between each pair of cities

w0=[999,454,317,165,528,222,223,410]
w1=[453,999,253,291,210,325,234,121]
w2=[317,252,999,202,226,108,158,140]
w3=[165,292,201,999,344,94,124,248]
w4=[508,210,235,346,999,336,303,94]
w5=[222,325,116,93,340,999,182,247]
w6=[223,235,158,125,302,185,999,206]
w7=[410,121,141,248,93,242,199,999]

distances = {0:w0,1:w1,2:w2,3:w3,4:w4,5:w5,6:w6,7:w7}


if __name__ == "__main__":

    # Constant that is an instance object 
    genetic_problem_instances = 10
    print("EXECUTING ", genetic_problem_instances, " INSTANCES \n")
    TSP(genetic_problem_instances)
