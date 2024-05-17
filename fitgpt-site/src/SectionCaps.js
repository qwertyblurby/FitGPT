import Floater from './Floater';

function SectionCaps() {
	const floaters = [
		[
			"Blazing Fast",
			"Our AI model processes images in milliseconds, delivering lightning-fast results.",
			"BoltIcon"
		],
		[
			"Efficient Model Architecture",
			"Our AI model is built on a highly optimized and efficient architecture, allowing for fast processing and deployment on a wide range of hardware.",
			"MinimizeIcon"
		],
		[
			"Detailed Predictions",
			"Our AI model provides detailed probability distributions for each article of clothing, giving you a comprehensive understanding of the classification results.",
			"ShirtIcon"
		],
		[
			"Automatic Person Detection",
			"Our advanced computer vision algorithms can automatically detect and segment people within images, enabling more accurate classification.",
			"PersonStandingIcon"
		]
	]
	
	return (
		<div className="space-y-6">
			<h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">Capabilities</h2>
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
