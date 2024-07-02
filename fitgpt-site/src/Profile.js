import GitHubIcon from './assets/icons/GitHubIcon';

function Profile({ name, bio, imageSrc, ghSrc }) {
	return (
		<div className="flex flex-row items-center mt-6 gap-4">
			<img
				alt={name}
				className="aspect-square rounded-full object-cover"
				height="64"
				width="64"
				src={imageSrc.includes("http") ? imageSrc : require(`./assets/about/${imageSrc}`)}
			/>
			<div className="text-gray-400">
				<h3 className="text-lg text-white font-semibold">
					{name}
					{ghSrc !== undefined && ghSrc !== "" && (
						<a className="hover:text-sky-300" href={ghSrc}>
							<GitHubIcon isProfile={true}/>
						</a>
					)}
				</h3>
				{bio}
			</div>
		</div>
	);
}

export default Profile;
