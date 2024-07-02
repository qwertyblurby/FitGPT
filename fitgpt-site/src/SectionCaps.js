import Floater from './Floater';

function SectionCaps() {
	const floaters = [
		[
			"Blazing Fast",
			"FitGPT processes images in seconds, delivering lightning-fast results.",
			"BoltIcon"
		],
		[
			"Robust Architecture",
			"Tried-and-true computer vision techniques enable FitGPT to produce reliable predictions while keeping parameter counts low.",
			"MinimizeIcon"
		],
		[
			"Detailed Predictions",
			"Probability distributions for each article of clothing give you insights into the model's confidence.",
			"ShirtIcon"
		],
		[
			"Automatic Person Detection",
			"FitGPT uses computer vision algorithms to automatically locate people within images, telling the model where to look.",
			"PersonStandingIcon"
		]
	]
	
	return (
		<div className="space-y-6">
			<h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">Model Capabilities</h2>
			<p className="text-gray-400">Scroll to our demo to see it for yourself!</p>
			<div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
				{floaters.map(([titleText, description, iconSrc]) => (
					<Floater
						key={titleText}
						titleText={titleText}
						description={description}
						iconSrc={iconSrc}
					/>
				))}
			</div>
		</div>
	)
}

export default SectionCaps;
