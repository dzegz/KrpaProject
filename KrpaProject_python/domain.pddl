(define(domain first_try)
	(:requirements :strips :typing)
	(:types gripper point)
	(:predicates 
		(gripping ?g - gripper ?p - point)
		(released ?g - gripper)
		(equal ?p1 ?p2 - point)
	)
	(:action grip 
		:parameters (?g - gripper ?p - point)
		:precondition (and (released ?g)) 
		:effect (and (not (released ?g)) (gripping ?g ?p))
    )
	(:action release
	    :parameters (?g - gripper ?p - point)
	    :precondition (and (not (released ?g)) (gripping ?g ?p))
	    :effect (and (released ?g) (not (gripping ?g ?p)))
	)
	(:action p2p 
	    :parameters (?g - gripper ?p1 ?p2 - point)
	    :precondition (and (not (released ?g)) (gripping ?g ?p1) (not (equal ?p1 ?p2)))
	    :effect (equal ?p1 ?p2)
	)
)

